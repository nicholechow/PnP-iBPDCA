import logging
from datetime import datetime
import torch
import numpy as np
from prox_PnP_restoration import PnP_restoration
from argparse import ArgumentParser
import utils.utils_image as util
from piq import fsim
import os

noise_percentage = 5
extrapolation = True
sigma = noise_percentage/100
assert sigma > 0, "sigma should be greater than 0"
image_dir = "../testsets/brainweb/t1"  # 'brain4' | 'TEST'
output_dir = os.path.join('../denoising', 'PnP_iBPDCA',
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
    noise1 = np.random.normal(0, sigma, img_L.shape)
    noise2 = np.random.normal(0, sigma, img_L.shape)
    img_L = np.sqrt((img_L + noise1)**2 + noise2**2)
    return img_L


def read_img_rician(img_dir, sigma):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_H = util.imread_uint(img_dir, n_channels=1)
    img_H = util.uint2double(img_H)
    img_L = np.copy(img_H)
    img_L = add_rician_noise(img_L, sigma)
    img_H = torch.from_numpy(img_H).to(device).permute(2, 0, 1)
    img_L = torch.from_numpy(img_L).to(device).permute(2, 0, 1)
    return img_L, img_H


# tunable parameters
delta = 0.9999
eta = 0.5  # fixed due to the paper Prxo-GS denoiser, which fix L_g <1, so max_eta = 0.5
# 10% noise: sigma= 25.5, Lambda= 1.3, gamma * 0.75, max extrapolation: 0.7071
# 5% noise: sigma= 12.75, Lambda= 1.3, gamma * 0.2, max extrapolation: 0.3055
# 3% noise: sigma= 7.65, Lambda= 1.6, gamma * 0.085, max extrapolation: 0.2381
# 1% noise: sigma= 2.55, Lambda= 1.9 gamma * 0.027, max extrapolation: 0.0949
if sigma == 0.1:
    c = 0.75
    Lambda = 1.3
    if extrapolation:
        beta = 0.7
    else :
        beta = 0.0
elif sigma == 0.05:
    c = 0.2
    Lambda = 1.3
    if extrapolation:
        beta = 0.36
    else:
        beta = 0.0
elif sigma == 0.03:
    c = 0.085
    Lambda = 1.6
    if extrapolation:
        beta = 0.24
    else:
        beta = 0.0
elif sigma == 0.01:
    c = 0.014
    Lambda = 1.9
    if extrapolation:
        beta = 0.1
    else:
        beta = 0.0
else:
    raise ValueError("sigma value not supported")
gamma = min((sigma*255)**2, 1/(delta+eta)) * c
# if (sigma*255)**2 > 1/(delta+eta):
#     print("1/(delta+eta) is smaller") 
tol = 1e-5
print(f"max extrapolation: {np.sqrt(gamma)}")
gamma = gamma/255
max_DCA_iter = 1000

############## Configure logger ##############
logger.info(f"gamma = {gamma}")
logger.info(f"Lambda = {Lambda}")
logger.info(f"sigma = {sigma}")
logger.info(f"tol = {tol}")
logger.info(f"beta = {beta}")
##############################################
parser = ArgumentParser()
parser = PnP_restoration.add_specific_args(parser)
hparams = parser.parse_args()
hparams.degradation_mode = 'denoising'
hparams.noise_level_img = np.sqrt(Lambda*gamma)
PnP_module = PnP_restoration(hparams)

def main(file_name):
    f, img_H = read_img_rician(file_name, sigma)
    un = f
    prev_un = un
    for oit in range(max_DCA_iter):
        temp = un
        un = un + beta * (un - prev_un)
        prev_un = temp
        z = (f*un)/(sigma**2)
        I0z = torch.special.i0e(z)
        I1z = torch.special.i1e(z)
        Iz = torch.div(I1z, I0z)
        vk = (f/(sigma**2))*Iz
        ########################## Plug-and-Play##########################
        tn = un - (gamma / sigma**2) * un + gamma * vk
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
        psnr_val = util.calculate_psnr(util.tensor2uint(un), util.tensor2uint(img_H))
        util.imsave(util.tensor2uint(un), "output.png")
        if diff < tol or oit == max_DCA_iter-1:
            ssim_val = util.calculate_ssim(
                util.tensor2uint(un), util.tensor2uint(img_H))
            util.imsave(util.tensor2uint(un), os.path.join(
                output_folder, f"{file_name.split('/')[-1]}"))
            fsim_val = fsim(torch.clamp(un,0,1).repeat(3, 1, 1).unsqueeze(0),img_H.repeat(3, 1, 1).unsqueeze(0), data_range=1.).item()
            return psnr_val, ssim_val, fsim_val, oit


if __name__ == '__main__':
    psnr_val, ssim_val, fsim_val , oit = main(os.path.join(image_dir, "t1_ai_msles2_1mm_pn0_rf0_axial_101.jpg"))
    print(f"PSNR: {psnr_val}, SSIM: {ssim_val}, FSIM: {fsim_val}, oit: {oit}")
    # psnr_list = []
    # ssim_list = []
    # fsim_list = []
    # n_iter_list = []
    # supported_extensions = [".png", ".jpg", ".jpeg", ".gif"]

    # file_list = sorted(os.listdir(image_dir))
    # i = 1
    # for filename in file_list:
    #     if any(filename.lower().endswith(ext) for ext in supported_extensions):
    #         psnr_val, ssim_val, fsim_val, n_iter = main(
    #             os.path.join(image_dir, filename))
    #         logger.info("---{}--> {} | {:.2f}dB; iter: {}".format(i, filename, psnr_val, n_iter))
    #         psnr_list.append(psnr_val)
    #         ssim_list.append(ssim_val)
    #         fsim_list.append(fsim_val)
    #         n_iter_list.append(n_iter)
    #         i += 1

    # logger.info(f"PSNR = {psnr_list}")
    # logger.info(f"SSIM = {ssim_list}")
    # logger.info(f"FSIM = {fsim_list}")
    # logger.info(f"n_iter = {n_iter_list}")
    # logger.info(f"Average PSNR = {np.mean(psnr_list)}")
    # logger.info(f"Average SSIM = {np.mean(ssim_list)}")
    # logger.info(f"Average FSIM = {np.mean(fsim_list)}")
    # logger.info(f"Average n_iter = {np.mean(n_iter_list)}")