import logging
from datetime import datetime
import torch
import numpy as np
from prox_PnP_restoration import PnP_restoration
from argparse import ArgumentParser
import utils.utils_image as util
from scipy.io import loadmat
from utils.utils_psf import *
from utils.utils_matlabengine import Au, Atu
from utils.utils_sisr import *
from piq import fsim
import os
import time
from scipy.io import savemat

noise_percentage = 5
sigma = noise_percentage/100
assert sigma > 0, "sigma should be greater than 0"
image_dir = "../testsets/brainweb/t2"  # 'brain4' | 'TEST'
output_dir = os.path.join('../denoising', 'PnP_BPDCA',
                          image_dir.split('/')[-1], str(sigma*255))

output_folder = os.path.join(output_dir, 'results')
log_folder = os.path.join(output_dir, 'log')
os.makedirs(output_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"{log_folder}/run_{current_time}.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def add_rician_noise(img_L, sigma):
    np.random.seed(seed=0)  # for reproducibility
    shape_img = img_L.transpose(1, 0, 2).shape
    noise1 = np.random.normal(0, sigma, shape_img).transpose(1, 0, 2)
    noise2 = np.random.normal(0, sigma, shape_img).transpose(1, 0, 2)
    img_L = np.sqrt((img_L + noise1)**2 + noise2**2)
    return img_L

def read_img_rician(img_dir, sigma):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_H = util.imread_uint(img_dir, n_channels=1)
    img_H = util.uint2double(img_H)
    
    A = loadmat('masks/gaussain_9_1.mat')['A'] 
    # img_L = cv2.filter2D(np.transpose(img_L,(2,0,1)), -1, A)
    img_L = np.squeeze(img_H)
    img_L = scipy.ndimage.correlate(img_L, A, mode='constant', origin=0)
    img_L = np.expand_dims(img_L, axis=2)
    img_L = add_rician_noise(img_L, sigma)
    img_L = torch.from_numpy(img_L).to(device).permute(2, 0, 1)
    img_H = torch.from_numpy(img_H).to(device).permute(2, 0, 1)

    # obtaining blur matrix
    [Nc, Nx, Ny] = img_H.shape
    A = torch.from_numpy(np.array(A)).to(device)
    point = (5, 5)
    center = tuple(1 - coord for coord in point)
    blur_A = padPSF(A, Nx, Ny)
    circshift = torch.roll(blur_A, center, dims=(0, 1)).to(A.device)
    blur_matrix_trans = torch.fft.fft2(circshift, dim=(-2, -1))
    return img_L, img_H, blur_matrix_trans

# tunable parameters
# 10% noise: sigma= 25.5, Lambda= 1.3, 0.2
# 5% noise: sigma= 12.75, Lambda= 1.3, 0.2
# 3% noise: sigma= 7.65, Lambda= 1.6, 0.25
# 1% noise: sigma= 2.55, Lambda= 1.9 0.35
gamma = (sigma**2) * 0.25
tol = 1e-4  # AI: 1e-4, CAI: 1e-5, element: 1e-4
Lambda = 1.1
max_DCA_iter = 1000

############## Configure logger ##############
logger.info(f"gamma = {gamma}")
logger.info(f"Lambda = {Lambda}")
logger.info(f"sigma = {sigma}")
logger.info(f"tol = {tol}")
##############################################
parser = ArgumentParser()
parser = PnP_restoration.add_specific_args(parser)
hparams = parser.parse_args()
hparams.degradation_mode = 'denoising'
hparams.noise_level_img = np.sqrt(Lambda*gamma)
PnP_module = PnP_restoration(hparams)


def main(file_name):
    f, img_H, blur_matrix_trans = read_img_rician(file_name, sigma)
    conj = torch.conj(blur_matrix_trans)
    un = f
    psnr_arr = []
    time_arr = []
    begin_time = time.time()
    for oit in range(max_DCA_iter):
        prev_un = un
        # z = (f*un)/(sigma**2)
        z = (f*Au(un, blur_matrix_trans))/(sigma**2)
        I0z = torch.special.i0e(z)
        I1z = torch.special.i1e(z)
        Iz = torch.div(I1z, I0z)
        # vk = (f/(sigma**2))*Iz
        vk = Au((f*Iz)/(sigma**2), blur_matrix_trans)
        ########################## Plug-and-Play##########################
        tn = un - (gamma / sigma**2) * Atu(Au(un,blur_matrix_trans),conj) + gamma * vk
        noise_im_tensor = tn.unsqueeze(0)
        noise_im_tensor = noise_im_tensor.repeat(1, 3, 1, 1)
        if hparams.grad_matching:
            # Calculate grad
            Dg, f_pnp = PnP_module.denoiser_model.calculate_grad(
                noise_im_tensor, hparams.noise_level_img)
            Dg = Dg.detach()
            f_pnp = f_pnp.detach()
            # Denoise
            denoise_img_tensor = noise_im_tensor - \
                PnP_module.denoiser_model.hparams.weight_Ds * Dg
        else:
            # Denoise
            denoise_img_tensor = PnP_module.denoiser_model.student_grad.forward(
                noise_im_tensor, hparams.noise_level_img)
        un = denoise_img_tensor.squeeze(0)
        un = torch.mean(un, dim=0, keepdim=True)
        ################### output###################
        diff = torch.norm(prev_un-un)/torch.norm(prev_un)
        psnr_val = util.calculate_psnr(
            util.tensor2uint(un), util.tensor2uint(img_H))
        psnr_arr.append(psnr_val)
        time_arr.append(time.time() - begin_time)
        if diff < tol or oit == max_DCA_iter-1:
            end_time = time.time()
            ssim_val = util.calculate_ssim(
                util.tensor2uint(un), util.tensor2uint(img_H))
            util.imsave(util.tensor2uint(un), os.path.join(
                output_folder, f"{file_name.split('/')[-1]}"))
            fsim_val = fsim(torch.clamp(un, 0, 1).repeat(3, 1, 1).unsqueeze(
                0), img_H.repeat(3, 1, 1).unsqueeze(0), data_range=1.).item()
            # import matplotlib.pyplot as plt
            # # Create a plot
            # plt.figure()
            # plt.plot(range(len(psnr_arr)), psnr_arr, marker='o')
            # plt.title('Normal')
            # plt.xlabel('Iter')
            # plt.ylabel('PSNR')
            # plt.grid(True)
            # # Save the plot as an image file
            # plt.savefig(os.path.join(output_folder,"_psnr_plot.png"))
            return psnr_val, ssim_val, fsim_val, oit, end_time - begin_time, psnr_arr, time_arr


if __name__ == '__main__':
    file_name ="t2_ai_msles2_1mm_pn0_rf0_axial_091.jpg"
    psnr_val, ssim_val, fsim_val , oit, time_taken, psnr_list, time_list = main(os.path.join(image_dir, file_name))
    logger.info(f"no PSNR = {psnr_val}")  
    logger.info(f"no SSIM = {ssim_val}")
    logger.info(f"no FSIM = {fsim_val}")
    # logger.info(f"n_iter = {oit}")
    logger.info(f"no time = {time_taken}")

    psnr_list_name = 'psnr_no' 
    time_list_name = 'time_no'
    data = {
        psnr_list_name: np.array(psnr_list),
        time_list_name: np.array(time_list)
    }
    file_name_without_ext = os.path.splitext(file_name)[0]
    filename = f"run_BPDCA_sigma_{sigma*255}_{file_name_without_ext}.mat"
    savemat(filename, data)
    # psnr_list = []
    # ssim_list = []
    # fsim_list = []
    # n_iter_list = []
    # time_list = []
    # supported_extensions = [".png", ".jpg", ".jpeg", ".gif"]

    # file_list = sorted(os.listdir(image_dir))
    # i = 1
    # for filename in file_list:
    #     if any(filename.lower().endswith(ext) for ext in supported_extensions):
    #         psnr_val, ssim_val, fsim_val, n_iter, time_rec, psnr_arr, time_arr = main(
    #             os.path.join(image_dir, filename))
    #         logger.info("---{}--> {} | {:.2f}dB; iter: {}".format(i,
    #                     filename, psnr_val, n_iter))
    #         psnr_list.append(psnr_val)
    #         ssim_list.append(ssim_val)
    #         fsim_list.append(fsim_val)
    #         n_iter_list.append(n_iter)
    #         time_list.append(time_rec)
    #         i += 1

    # # logger.info(f"PSNR = {psnr_list}")
    # # logger.info(f"SSIM = {ssim_list}")
    # # logger.info(f"FSIM = {fsim_list}")
    # # logger.info(f"n_iter = {n_iter_list}")
    # logger.info(f"Average PSNR = {np.mean(psnr_list)}")
    # logger.info(f"Average SSIM = {np.mean(ssim_list)}")
    # logger.info(f"Average FSIM = {np.mean(fsim_list)}")
    # # logger.info(f"Average n_iter = {np.mean(n_iter_list)}")
    # logger.info(f"Average time = {np.mean(time_list)}")
