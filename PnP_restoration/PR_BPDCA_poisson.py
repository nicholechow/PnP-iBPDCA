import torch
import utils.utils_image as util
import numpy as np
from prox_PnP_restoration import PnP_restoration
import logging
from datetime import datetime
from argparse import ArgumentParser
import utils.utils_image as util
import os
from scipy.io import loadmat
import time
from piq import fsim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_DCA_iter = 1000
# best param:
# L =4
# alpha = 9, constant = 0.5, tol= 1e-4
# alpha = 27, constant = 0.5 , tol = 1e-3
# alpha = 81, constant = 0.56, tol = 1e-3
alpha = 9
if alpha == 9:
    constant = 0.51
    tol = 1e-4
elif alpha == 27:
    constant = 0.5
    tol = 1e-3
elif alpha == 81:
    constant = 0.55
    tol = 1e-3
choice_scale = 0.99

image_dir = "../testsets/PrDeep_12"  # 'brain4' | 'TEST'
output_dir = os.path.join('../PR_poisson',image_dir.split('/')[-1], str(alpha))

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

logger.info(f"alpha = {alpha}")
logger.info(f"tol = {tol}")
logger.info(f"constant = {constant}")


def main(img_dir):
    img_H = util.imread_uint(img_dir, n_channels=1)
    img_H = util.uint2double(img_H)
    img_H = torch.from_numpy(img_H).to(device).permute(2, 0, 1).unsqueeze(0)
    n1 = img_H.shape[2]
    n2 = img_H.shape[3]
    # Masks = create_cdp_masks('complex',n1,n2)

    Masks = loadmat('masks/mask_4_128.mat')['mask'] 
    L = Masks.shape[0]
    Masks = torch.from_numpy(Masks).to(device).unsqueeze(0)

    def Poisson_noise_torch(z, alpha):
        z2 = z**2
        torch.manual_seed(0)
        intensity_noise = (alpha/255) * torch.abs(z) * torch.randn_like(z)
        y2 = torch.clamp(z2 + intensity_noise, min=0)
        y = torch.sqrt(y2)
        rr =  y - torch.abs(z)
        sigma = rr.std()
        return y, sigma

    def A(x, SamplingRate= L, mask = Masks, device = device):
        '''
        CDP measurement matrix
        x: batch data shape: (batch, 1, imsize, imsize)
        mask: uniform masks shape: (1, SamplingRate, imsize, imsize)
        '''
        imsize = x.shape[2]   # 128
        # batch_size,SamplingRate,imsize,imsize
        x = x.repeat(1, SamplingRate, 1, 1)
        x = mask*x
        Ax = torch.zeros_like(x).to(device)   # batchsize*4*128*128 complex
        for i in range(SamplingRate):
            Ax[:, i, :, :] = torch.fft.fft2(
                x[:, i, :, :]) * (1/imsize)   # 128*128_complex
        return Ax   # batchsize*4*128*128_complex
    
    def At(Ax, SamplingRate = L, mask = Masks):
        '''
        CDP measurement inverse matrix
        Ax: OverSampled Fourier Transform of original data shape:(batch_size, SamplingRate, imsize, imsize)

        '''
        B, C, imsize1, imsize2 = Ax.shape  # 128
        Atx = torch.zeros_like(Ax)  # batch_size*SamplingRate*128*128
        for i in range(SamplingRate):
            Atx[:, i, :, :] = torch.fft.ifft2(Ax[:, i, :, :])   # 128*128
        mask_ = torch.conj(mask)   # batch_size * SamplingRate * 128 * 128_complex
        # batch_size * 1 * 128 * 128_complex
        Atx = torch.sum(mask_ * Atx, axis=1) * imsize1
        return torch.real(Atx).reshape(B, 1, imsize1, imsize2)  # 128*128
    
    YGauss, sigma = Poisson_noise_torch(torch.abs(A(img_H)), alpha=alpha)

    grad_2_f1 = torch.real(torch.fft.ifft2(Masks) * torch.fft.fft2(Masks) * torch.abs(A(img_H)**2))
    largest_eigenvalue = torch.max(torch.linalg.eigvals(grad_2_f1).real)
    eyes = torch.eye(n1,n2, device=device).unsqueeze(0).unsqueeze(0)
    grad_2_h = (torch.norm(img_H)**2 +1) * eyes + 2 * img_H * torch.transpose(img_H, 2, 3)
    smallest_eigenvalue = torch.min(torch.linalg.eigvals(grad_2_h).real)
    Lambda = (smallest_eigenvalue/largest_eigenvalue) * choice_scale

    z = torch.ones(1, 1, n1, n2, device=device) 

    def grad_f2(z):
        return At(A(z)* YGauss**2)

    def grad_f1(z):
        return At(A(z)*torch.abs(A(z))**2)

    def grad_h(z):
        return (torch.norm(z)**2 + 1) * z

    def grad_conj_h(z):
        norm = torch.norm(z)**2
        coeff = [norm.cpu(), 0, 1, -1]
        roots = np.roots(coeff)
        positive_real_roots = [root for root in roots if root.real > 0]

        # Check if there is exactly one positive real root
        if len(positive_real_roots) != 1:
            raise ValueError(
                "The positive real root is not unique or doesn't exist")
        positive_real_root = np.real(positive_real_roots[0])
        return positive_real_root * z

    parser = ArgumentParser()
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()
    hparams.degradation_mode = 'denoising'
    hparams.noise_level_img = sigma * constant
    PnP_module = PnP_restoration(hparams)
    time_start = time.time()
    for oit in range(1000):
        prev_z = z
        z = grad_conj_h(
        grad_h(z) - Lambda *(grad_f1(z) - grad_f2(z)))
        noise_im_tensor = z.repeat(1, 3, 1, 1)
        # ##############################
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
        z = denoise_img_tensor.squeeze(0)
        z = torch.mean(z, dim=0, keepdim=True)
        ################### output###################
        diff = torch.norm(prev_z-z)/torch.norm(prev_z)
        psnr_val = util.calculate_psnr(util.tensor2uint(z), util.tensor2uint(img_H))
        # print(f"iter: {oit}, PSNR: {psnr_val}, diff: {diff:.4e}")
        if diff < tol or oit == max_DCA_iter-1:
            time_end = time.time()
            ssim_val = util.calculate_ssim(
                util.tensor2uint(z), util.tensor2uint(img_H))
            fsim_val = fsim(torch.clamp(z,0,1).repeat(3, 1, 1).unsqueeze(0),img_H.repeat(1,3, 1, 1), data_range=1.).item()
            util.imsave(util.tensor2uint(z), os.path.join(
                output_folder, f"{filename.split('/')[-1]}"))
            return psnr_val, ssim_val, oit, time_end - time_start, fsim_val
    
if __name__ == '__main__':
    psnr_list = []
    ssim_list = []
    fsim_list = []
    n_iter_list = []
    snr_list = []
    runtime_list = []
    supported_extensions = [".png", ".jpg", ".jpeg", ".gif", ".tif"]

    file_list = sorted(os.listdir(image_dir))
    i = 1
    for filename in file_list:
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            psnr_val, ssim_val, n_iter, runtime, fsim_val = main(
                os.path.join(image_dir, filename))
            logger.info("---{}--> {} | {:.2f}dB; SSIM: {:.4f}".format(i, filename, psnr_val, ssim_val))
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            n_iter_list.append(n_iter)
            runtime_list.append(runtime)
            fsim_list.append(fsim_val)
            i += 1

    logger.info(f"PSNR = {psnr_list}")
    logger.info(f"SSIM = {ssim_list}")
    logger.info(f"n_iter = {n_iter_list}")
    logger.info(f"Average PSNR = {np.mean(psnr_list)}")
    logger.info(f"Average SSIM = {np.mean(ssim_list)}")
    logger.info(f"Average n_iter = {np.mean(n_iter_list)}")
    logger.info(f"Average runtime = {np.mean(runtime_list)}")
    logger.info(f"Avergae FSIM = {np.mean(fsim_list)}")