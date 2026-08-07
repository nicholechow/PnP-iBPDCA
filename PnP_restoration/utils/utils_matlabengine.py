import numpy as np
import torch


def divergence(Px, Py):
    fx = Px - torch.cat((Px[:, 0, :].unsqueeze(1), Px[:, :-1, :]), dim=1)
    fx[:, 0, :] = Px[:, 0, :]
    fx[:, -1, :] = -Px[:, -2, :]

    fy = Py - torch.cat((Py[:, :, 0].unsqueeze(2), Py[:, :, :-1]), dim=2)
    fy[:, :, 0] = Py[:, :, 0]
    fy[:, :, -1] = -Py[:, :, -2]
    return fx + fy


def gradient(M):
    fx = torch.cat((M[:, 1:, :], M[:, -1, :].unsqueeze(1)), dim=1) - M
    fy = torch.cat((M[:, :, 1:], M[:, :, -1].unsqueeze(2)), dim=2) - M
    return fx, fy


def trans(X):
    if X.dim() == 2:
        [Nx, Ny] = X.shape
    else:
        [Nc, Nx, Ny] = X.shape
    return torch.fft.fft2(X, dim=(-2, -1)) * (1/np.sqrt(Ny*Nx))


def itrans(X):
    if X.dim() == 2:
        [Nx, Ny] = X.shape
    else:
        [Nc, Nx, Ny] = X.shape
    return torch.fft.ifft2(X, dim=(-2, -1)) * (np.sqrt(Ny*Nx))


def Au(u, blur_matrix_trans):
    return torch.real(itrans(blur_matrix_trans*trans(u)))


def Atu(u, conj):
    return torch.real(itrans(trans(u)*conj))


def loss_PnP(un, sigma, beta, gn, tn, I0z, blur_matrix_trans):
    return 1/(2*sigma**2)*(torch.norm(Au(un, blur_matrix_trans))**2) - torch.sum(torch.sum(torch.log(I0z))) + gn - torch.norm(gn-tn)**2/(2*beta)


def loss_PDCA(un, sigma, ugrad, gamma, I0z, blur_matrix_trans):
    return 1/(2*sigma**2)*(torch.norm(Au(un, blur_matrix_trans))**2) - torch.sum(torch.sum(torch.log(I0z))) + gamma * torch.sum(torch.sum(ugrad))


def read_img(img_dir, sigma):
    import matlab.engine
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not isinstance(sigma, float):
        sigma = float(sigma)
    eng = matlab.engine.start_matlab()
    s = eng.genpath('./')
    eng.addpath(s, nargout=0)
    f0 = eng.double(eng.im2gray(eng.imread(img_dir)))
    OG = torch.from_numpy(np.array(f0)).unsqueeze(0).to(device)
    A = eng.fspecial('motion', 5.0, 30.0)
    f0 = eng.imfilter(f0, A)
    f = eng.ricernd(f0, sigma)
    # eng.imshow(eng.rdivide(f,255.),nargout=0)
    f0 = torch.from_numpy(np.array(f0)).unsqueeze(0).to(device)
    f = torch.from_numpy(np.array(f)).unsqueeze(0).to(device)
    A = torch.from_numpy(np.array(A)).to(device)
    eng.quit()
    return f, f0, A, OG
