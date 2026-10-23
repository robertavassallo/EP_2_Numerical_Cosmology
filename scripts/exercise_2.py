from __future__ import annotations

import numpy as np
from scipy.stats import binned_statistic

from exercise_3 import (Pk_powerlaw, gaussian_field, kgrid, derivative_k,
                        real_space_field, fourier_space_field)

# ------------------------- second-order SPT density --------------------------------

def f2_kernel(k1: np.ndarray, k2: np.ndarray):
    """Symmetric EdS kernel F2(k1, k2) = 5/7 + mu/2 (k1/k2 + k2/k1) + 2/7 mu^2, where mu = k1_hat k2_hat
    Inputs: k1, k2 wavevectors, (arrays of shape (M, 3))
    Returns: F2 (array of shape (M,))"""
    mu = np.sum(k1 * k2, axis=-1) / (np.linalg.norm(k1, axis=-1) * np.linalg.norm(k2, axis=-1))
    return 5/7 + 0.5 * mu * (np.linalg.norm(k1, axis=-1) / np.linalg.norm(k2, axis=-1) + np.linalg.norm(k2, axis=-1) / np.linalg.norm(k1, axis=-1)) + 2/7 * mu**2
    
def band_limit_mask(N: int):
    """Mask of shape (N, N, N) to select integer mode indices n_i such that |n_i| < N/3 on each axis"""
    n = np.rint(np.fft.fftfreq(N) * N).astype(int) # norm of the number of wavelenghts that fix the box rounded to nearest integer (mode indices)
    keep = 3 * np.abs(n) < N   # cut to 2/3 of box size in each direction
    return keep[:, None, None] & keep[None, :, None] & keep[None, None, :]

def delta_2(delta1_k: np.ndarray, N: int, L: float):
    """delta_{2,k} from delta_{1,k} by transforming to real space and evaluating the SPT expression
    delta_2(x) = 5/7 delta1^2 + del_i delta1 del_i phi + 2/7 (del_i del_j phi)(del_i del_j phi), where laplacian(phi) = delta1"""
    mask = band_limit_mask(N)
    delta1k = delta1_k * mask # band-limited copy

    ks = [derivative_k(N, L, axis) for axis in range(3)] # kx, ky, kz with Nyquist mode set to 0
    *_, k_mag = kgrid(N, L)
    k2 = k_mag**2
    phi_k = np.where(k2 == 0, 0.0, -delta1k / np.where(k2 == 0, 1.0, k2))  # phi_k = -delta_k/k^2, 0 at k = 0

    delta1x = real_space_field(delta1k, N, L)[0] # real space linear density field
    delta2 = 5/7 * delta1x**2 # first term of delta_2(x) 
    for i in range(3):
        del_i_delta1 = real_space_field(1j * ks[i] * delta1k, N, L)[0] # ik_i delta1_k --> del_i delta1(x)
        del_i_phi = real_space_field(1j * ks[i] * phi_k, N, L)[0] # ik_i phi(k) --> del_i phi(x)
        delta2 += del_i_delta1 * del_i_phi 
        for j in range(3):
            delta2 += 2/7 * real_space_field(-ks[i] * ks[j] * phi_k, N, L)[0]**2 # -k_i k_j phi(k) --> del_i del_j phi(x)

    delta2_k = fourier_space_field(delta2, N, L)
    zero_mode = delta2_k[0, 0, 0] # zero mode of delta2_k
    delta2_k = delta2_k * mask
    delta2_k[0, 0, 0] = 0.0
    return delta2_k, zero_mode

# -------------------- Bispectrum ----------------------------------

def power_spectrum(delta_k: np.ndarray, N: int, L: float, nbins: int = 20):
    """P_hat(k) = 1/(N_k V) sum_{k_a in S_k} |delta_k|^2 in nbins linear shells from k_f=2pi/L to k_max = ((N-1)//3) k_f
    Returns (k, P(k), N(k)), where k is the mean k of the modes in each shell, P(k) the mean power and N(k) the number of modes"""
    *_, k_mag = kgrid(N, L)
    k_f = 2*np.pi / L
    k_max = ((N - 1) // 3) * k_f # keeps modes where |k| < (2/3) k_Ny
    bins = np.linspace(k_f, k_max, nbins + 1)
    power = np.abs(delta_k)**2 / L**3  # |delta_k|^2 / V in each cell
    k = k_mag.ravel()
    k_mean = binned_statistic(k, k, statistic='mean', bins=bins)[0] # mean k of the modes in each shell
    Pk = binned_statistic(k, power.ravel(), statistic='mean', bins=bins)[0] # mean power in each shell
    nk = binned_statistic(k, power.ravel(), statistic='count', bins=bins)[0] # number of modes in each shell
    return k_mean, Pk, nk

def shell_mask(k_mag: np.ndarray, k_c: float, dk: float):
    """Mask that selects Fourier modes in the shell k_c - dk/2 <= |k| < k_c + dk/, 
    where k_c is the shell centre and dk the shell width in h/Mpc"""
    epsilon = 1e-9 * k_c  # shift for modes that lie on edges
    return (k_mag >= k_c - dk/2 - epsilon) & (k_mag < k_c + dk/2 - epsilon) & (k_mag > 0)

def bispectrum(delta_k: np.ndarray, masks: list, N: int, L: float):
    """B_hat of a bin of closed triangles with sides lying in the three shells defined
    B_hat = <d_s1 d_s2 d_s3>_x / (V <I_s1 I_s2 I_s3>_x),   N_T = V^3 <I_s1 I_s2 I_s3>_x
    where d_s(x) is delta filtered to the shell and I_s(x) the shell mask
    Returns: B_hat (float, in (Mpc/h)^6), N_T (number of closed triangles in the bin)"""
    V = L**3

    d_s = [real_space_field(delta_k * m, N, L)[0] for m in masks] # delta_k filtered to the shells transformed to real space
    I_s = [real_space_field(m, N, L)[0] for m in masks] # shell masks transformed to real space for triangle count

    # triangle sum: mean of the product of the 3 filtered fields over the grid cells
    triangles = np.mean(d_s[0] * d_s[1] * d_s[2])
    # triangle count: mean of the product of the 3 indicator (count) fields over the grid cells
    triangle_count = np.mean(I_s[0] * I_s[1] * I_s[2])

    return triangles / (V * triangle_count), V**3 * triangle_count  # B_hat = sum delta_1 delta_2 delta_3 / V = triangles/ (V * <I_1 I_2 I_3>), N_T = V^3 * <I_1 I_2 I_3>

def tree_bispectrum(masks: list, N: int, L: float, Pk):
    """B_tree averaged over the closed triangles of a bin
    B_tree(k1, k2, k3) = 2 F2(k1, k2) P(k1) P(k2) + 2 F2(k2, k3) P(k2) P(k3) + 2 F2(k3, k1) P(k3) P(k1)
    Inputs: masks: 3 shell masks, Pk: function of theoretical P_L(k) in (Mpc/h)^3 for k in h/Mpc
    Returns: B_tree averaged over the bin (float, in (Mpc/h)^6), number of triangles"""
    k_f = 2*np.pi / L

    n1d = np.rint(np.fft.fftfreq(N) * N).astype(int) # integer mode index of each grid index
    n_a_all = n1d[np.argwhere(masks[0])] # integer wavevectors (n_x, n_y, n_z), in units of k_f, of the modes in the first shell (N_k, 3)
    n_b_all = n1d[np.argwhere(masks[1])] # " " " in the second shell
    # argwhere returns the grid indices of True entries of each mask and n1d converts them to mode indices

    total, count = 0.0, 0
    for n_a in n_a_all:    # each k1 of first shell
        n_c = -(n_a + n_b_all)  # k3 = -(k1 + k2) for every k2 of the second shell
        in_shell = masks[2][tuple((n_c % N).T)] # only pairs whose k3 is inside third shell are kept (n mod N = grid index)

        k_a, k_b, k_c = k_f * n_a, k_f * n_b_all[in_shell], k_f * n_c[in_shell] # wavevectors
        P_a, P_b, P_c = (Pk(np.linalg.norm(k, axis=-1)) for k in (k_a, k_b, k_c)) # theoretical linear spectra for the 3 sides of the triangles
        B = 2*f2_kernel(k_a, k_b)*P_a*P_b + 2*f2_kernel(k_b, k_c)*P_b*P_c + 2*f2_kernel(k_c, k_a)*P_c*P_a # B_tree(k1, k2, k3) for all pairs with the given k1 

        total += B.sum(); count += in_shell.sum() 
    return total / count, count 
