"""
Builds a periodic cube of side L, N_g^3 mesh, V = L^3 in Fourier space
with gaussian density contrast field scaled by a P(k), delta(k) = sqrt(V P(k)) * (g1 + i g2)/sqrt(2) with g1, g2 ~ N(0, 1)
modes are k_i = 2 pi n_i / L,  for n_i in {0, 1, ..., N/2-1, -N/2, ..., -1}   (np.fft.fftfreq)

FFT-library conversion (numpy):
    np.fft.ifftn(a)[j] = (1/N^3) sum_n a[n] exp(2 pi i n.j/N)
    delta(x) = (1/V) sum_k delta_k e^{ik.x} = (N^3/V) * np.fft.ifftn(delta_k)
    delta_k = (V/N^3) * np.fft.fftn(delta(x))
"""
from __future__ import annotations

import numpy as np
import camb
import itertools
from scipy.stats import binned_statistic

# ----------------------Gaussian field generation--------------------------------
# k-space grid
def kgrid(N: int, L: float):
    """Returns k-values (kx, ky, kz, k_mag) in Fourier space grid in units of 1/[L]
    from each cell of the 3D grid"""
    k_grid = np.fft.fftfreq(N, d=L/N)*2*np.pi
    kx, ky, kz = np.meshgrid(k_grid, k_grid, k_grid, indexing="ij", sparse=True)
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    return kx, ky, kz, k_mag

def derivative_k(N: int, L: float, axis: int):
    """1D wavenumber array for first derivative along the given axis 
    with the Nyquist k set to zero"""
    k1d = 2*np.pi * np.fft.fftfreq(N, d=L/N)
    if N % 2 == 0:
        k1d[N // 2] = 0.0
    shape = [1, 1, 1]
    shape[axis] = N
    return k1d.reshape(shape)

def find_opposite(N: int):
    """Returns for each cell of the grid with wavevector k, 
    the index of the cell with -k"""
    ids = np.arange(N**3).reshape(N, N, N) #index of each cell
    m = (-np.arange(N)) % N #rest of division finds index of cell with opposite k
    return ids, ids[np.ix_(m, m, m)]

# Linear P(k) from CAMB
def Pk_CAMB(a_ref: float = 1.0, kmax: float = 10.0,
             H0=67.4, ombh2=0.0224, omch2=0.120, As=2.1e-9, ns=0.965, tau=0.054):
    """Returns the linear P(k) from CAMB at a_ref in (Mpc/h)^3 for k in h/Mpc"""
    z_ref = 1/a_ref - 1
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=0.0, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    pars.set_matter_power(redshifts=[z_ref], kmax=kmax)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    interp = results.get_matter_power_interpolator(nonlinear=False, hubble_units=True,
                                                   k_hunit=True, extrap_kmax=kmax * 10)
    kmin = 1e-4

    def Pk(k):
        return interp.P(z_ref, np.clip(k, kmin, None))
    return Pk

def Pk_powerlaw(A: float, ns: float = 0.965):
    """Linear P(k) = A k^(ns-3) for a_ref, where D1(a_ref) = 1
    Inputs:
        A: amplitude in (Mpc/h)^(3+ns)
        ns: power ('spectral index' if it were a primordial spectrum)
    Returns: function Pk(k) that takes k in h/Mpc and returns P in (Mpc/h)^3 at a_ref
    """
    def Pk(k):
        return A * np.asarray(k)**(ns-3)
    return Pk

#gaussian random field in Fourier space weighted by P(k)
def gaussian_field(Pk, N: int, L: float, rng: np.random.Generator):
    """Draws delta_k with <|delta_k|^2> = V P(k) and delta_{-k} = delta_k^*
    as a complex gaussian white noise with variance V P(k)"""
    V = L**3
    *_, k_mag = kgrid(N, L) # Fourier space grid
    amp = np.sqrt(V * Pk(np.where(k_mag == 0, 1.0, k_mag))) # amplitude of the gaussian random field (linear density field)

    g1 = rng.standard_normal((N, N, N))
    g2 = rng.standard_normal((N, N, N))
    delta_k = (amp/np.sqrt(2.0)) * (g1 + 1j*g2) # density field in Fourier space

    ids, partner = find_opposite(N) # pairs of cells with k and -k
    flat = delta_k.ravel()
    neg = (partner < ids).ravel() # mask to select element corresponding to -k of each pair
    flat[neg] = np.conj(flat[partner.ravel()[neg]]) # sets values of -k to be the complex conjugate of k

    selfc = (partner == ids) # mask to select cells that are their own -k pair (k == -k mod N)
    g3 = rng.standard_normal(selfc.sum()) # real gaussian random numbers for self-conjugate cells
    delta_k[selfc] = amp[selfc] * g3 
    delta_k[0, 0, 0] = 0.0                 # zero mode
    return delta_k  

def real_space_field(delta_k: np.ndarray, N: int, L: float):
    """delta(x) = (1/V) sum_k delta_k e^{ik.x} = (N^3/V) ifftn(delta_k)"""
    field = np.fft.ifftn(delta_k) * (N**3 / L**3)
    return field.real, np.abs(field.imag).max()

def fourier_space_field(field, N: int, L: float):
    """inverse of real_space_field: delta_k = (V/N^3) fftn(delta)"""
    return np.fft.fftn(field) * (L**3 / N**3)

def check_gaussian_field(delta_k, N: int, L: float, Pk, rng: np.random.Generator,
                         Ns: int = 16, nreal: int = 500):
    """Verification of Gaussian field: mean density, max|Im| consistent with
    round-off and delta_{-k} = delta_k^*. Also checks that self-conjugate cells are real with variance V P(k)"""
    V = L**3
    # Hermitian symmetry
    m = (-np.arange(N)) % N   # index of -k along each axis
    delta_mk = np.conj(delta_k[np.ix_(m, m, m)])  # delta_{-k}^* on the cell of k
    herm = np.abs(delta_k - delta_mk).max() / np.abs(delta_k).max()  # = 0 if delta_{-k} = delta_k^*

    # imaginary FFT residual and mean density
    delta_x, max_imag = real_space_field(delta_k, N, L)  # inverse FFT, max_imag = max|Im delta(x)|

    # check if self-conjugate cells are real with variance V P(k)
    *_, k_s = kgrid(Ns, L)
    ids_s, partner_s = find_opposite(Ns) #ids of cell with k and -k
    selfc_s = (partner_s == ids_s); selfc_s[0, 0, 0] = False   # mask for cells with k == -k, excluding zero mode
    var = V * Pk(k_s[selfc_s])  # expected <|delta_k|^2> on those cells
    g = np.stack([gaussian_field(Pk, Ns, L, rng)[selfc_s] for _ in range(nreal)]) # nreal draws of the gaussian field
    ratio = g.real**2 / var  # should be ~1 if self-conjugate cells are real with variance V P(k)
    sc_mean = ratio.mean()

    return {"zero_mode": abs(delta_k[0, 0, 0]),                # should be 0
            "mean_delta": delta_x.mean(),                      # mean density contrast, ~ 0
            "hermitian_rel_err": herm,                         # ~ 0 (up to round-off)
            "imag_rel_ifft": max_imag / np.abs(delta_x).max(), # ~1e-16 if Hermitian
            "imag_selfconj": np.abs(g.imag).max(),             # if 0, self-conjugate cells are real
            "selfconj_mean": sc_mean,                          # <|delta_k|^2 / (V P)> on those cells
            "selfconj_z": (sc_mean - 1.0) * np.sqrt(ratio.size / 2.0) # z score of the mean of |delta_k|^2 / (V P): (sc_mean − 1) / √(2/n)
            }

# --------------------- ZA positions and velocities ----------------------------
class Zeldovich:
    def __init__(self, delta_ref_k, N: int, L: float, cosmo, a_i: float = 0.02, a_ref: float = 1.0,
                 H0: float = 100.0):
        self.N, self.L = N, L

        ks = [derivative_k(N, L, axis) for axis in range(3)] #kx, ky, kz with Nyquist k set to zero
        self.ks, self.delta_ref_k = ks, delta_ref_k
        self.k2 = self.ks[0]**2 + self.ks[1]**2 + self.ks[2]**2 # k^2 for each cell

        # growth factor (D1(a_i) for D1(a_ref) = 1), growth rate (f(a_i)) and H(a_i)a_i
        D, f = cosmo.growing_mode(np.array([a_i, a_ref]), a_i=a_i)
        self.D1_i, self.f1_i = D[0]/D[1], f[0]
        self.H_a_i = a_i * H0 * cosmo.E(a_i)   # km/s per (Mpc/h), with H0 = 100 km/s per (Mpc/h)
    
    def displacement(self):
        # Psi_k = i D1 k / k^2 delta_ref,k
        zero = (self.k2 == 0.0)  # mask for cells where k^2 = 0 (k = 0 and Nyquist modes)
        k2 = np.where(zero, 1.0, self.k2)  # replaces k^2 = 0 with 1 to avoid division by zero 
        psi_k = [np.where(zero, 0.0, 1j * self.D1_i * k/k2 * self.delta_ref_k) for k in self.ks] # Psi_k = i D1 (k/k^2) delta_ref,k and Psi_k = 0 for k = 0 and Nyquist modes

        # Psi_p at particle labels q = (i, j, l) L/N in real space (N^3, 3)
        psi_p = np.stack([real_space_field(c, self.N, self.L)[0] for c in psi_k], axis=-1) # inverse Fourier transform of Psi_k --> (N, N, N, 3)
        psi_p -= psi_p.mean(axis=(0, 1, 2))  # mean displacement set to zero
        grid = np.arange(self.N) * (self.L/self.N) # 1D grid of particle labels q = (i, j, l) L/N in real space
        q = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1) # x,y,z coordinates of particle labels q = (i, j, l) L/N in real space
        q, psi = q.reshape(-1, 3), psi_p.reshape(-1, 3) #N^3 particles with 3 coordinates each (N^3, 3)

        # positions and velocities
        x = (q + psi) % self.L # periodic conditions
        v = self.H_a_i * self.f1_i * psi

        return psi, psi_k, x, v

# ------------------------- validation tests ----------------------------------
    def check(self):
        """1st part: velocity, displacement curl, divergence and mean checks with same discrete operators 
        used in the construction
        2nd part: checks if positions are wrapped in [0, L), particle number (mass) is unchanged
        and that center-of-mass shift is zero up to jumps of L in the periodic box"""
        psi_p, psi_k, x, v = self.displacement() 

        #displacement checks
        div = sum(real_space_field(1j * self.ks[a] * psi_k[a], self.N, self.L)[0] for a in range(3)) # divergence of Psi in real space: real_space(i k Psi_k) 
        D1_delta_ref = self.D1_i * real_space_field(np.where(self.k2 == 0.0, 0.0, self.delta_ref_k), self.N, self.L)[0] # D1 delta_ref in real space with cells where k^2 = 0 set to zero

        curl = [real_space_field(1j * (self.ks[(a+1) % 3] * psi_k[(a+2) % 3] - self.ks[(a+2) % 3] * psi_k[(a+1) % 3]),
                                 self.N, self.L)[0] for a in range(3)] #curl of Psi in real space: real_space(i k x Psi_k) for each axis

        mean_psi = np.array([real_space_field(c, self.N, self.L)[0].mean() for c in psi_k]) # mean displacement in each axis
        v_slope = np.vdot(psi_p, v) / np.vdot(psi_p, psi_p) # slope of v = H f Psi in km/s per (Mpc/h)

        #particle checks
        q = np.indices((self.N,) * 3).reshape(3, -1).T * (self.L / self.N)   # particle labels q = (i, j, l) L/N
        d = (x - q + self.L / 2) % self.L - self.L / 2  # displacement modulo in [-L/2, L/2)

        return {"mean_psi": mean_psi, #mean displacement in each axis in Mpc/h
                "div_rel_err": np.abs(-div - D1_delta_ref).max() / np.abs(D1_delta_ref).max(), #maximum relative error of div of Psi compared to D1 delta_ref
                "curl_rel":  max(np.abs(c).max() for c in curl) / np.abs(D1_delta_ref).max(), #maximum relative error of curl of Psi compared to max of D1 delta_ref
                "mean_psi_rel": np.abs(mean_psi).max() / psi_p.std(), # check if mean displacement is consistent with 0
                "v_norm": v_slope / (self.H_a_i * self.f1_i), # check if slope is consistent with H f
                "n_wrapped": int(np.sum(((q + psi_p) < 0) | ((q + psi_p) >= self.L))), # coordinates that crossed the box edge
                "in_box": bool(x.min() >= 0 and x.max() < self.L), # positions in [0, L)
                "Np_unchanged": bool(x.shape == q.shape and np.isfinite(x).all()), # number count conservation
                "com_shift": np.abs(d.mean(axis=0)).max() / psi_p.std()} # center-of-mass shift modulo L ~ 0 
    
# ------------------------- particle assignment and P(k) measurement -------------------------

class Pk_measurement:
    def __init__(self, delta_k, za_positions: Zeldovich, N: int, L: float, nbins: int = 20):
        self.N, self.L, self.nbins = N, L, nbins
        self.pos = za_positions.displacement()[2] # particle positions in real space
        self.Np = len(self.pos)  # number of particles
        self.delta_k_gauss = za_positions.D1_i * delta_k  # Gaussian field at a_i in Fourier space

        self.counts = self.density_cic() # particles per cell (mass in units of m_p)
        self.delta_part = (self.counts / (self.Np / self.N**3)) - 1.0 # density contrast of particles in each cell
        self.delta_k_part = fourier_space_field(self.delta_part, self.N, self.L) # density contrast of particles in Fourier space
        self.W = self.window()    

    #Cloud-in-Cell (CIC) assignment scheme
    def density_cic(self):
        """
        Assigns particle masses (with equal mass) to a 3D grid using the Cloud-in-Cell (CIC) scheme
        with periodic boundary conditions"""
        density_grid = np.zeros((self.N, self.N, self.N), dtype=np.float64)
        l = self.L/self.N
        N_grid = self.N
        x, y, z = self.pos[:, 0], self.pos[:, 1], self.pos[:, 2] 

        pos_x, pos_y, pos_z = x/l, y/l, z/l # fractional position (in units of cell size l)
        i, j, k = np.floor(pos_x).astype(int), np.floor(pos_y).astype(int), np.floor(pos_z).astype(int) # "home" cell index (integer part)
        
        tx, ty, tz = pos_x - i, pos_y - j, pos_z - k # fractional distance to the next cell boundary (weight)
        omx, omy, omz = 1.0 - tx, 1.0 - ty, 1.0 - tz # complementary weights for "home" cell
        
        i0, j0, k0 = i % N_grid, j % N_grid, k % N_grid # periodic conditions on "home" cell
        i1, j1, k1 = (i + 1) % N_grid, (j + 1) % N_grid, (k + 1) % N_grid # on neighboring cells (i+1, j+1, k+1)
        
        # mass assignment to the 8 surrounding cells of each particle    
        X, Y, Z = ((i0, omx), (i1, tx)), ((j0, omy), (j1, ty)), ((k0, omz), (k1, tz)) # (index, weight) of home and next cell for each axis
        corners = itertools.product(X, Y, Z) # 8 corner cells for each particle
        for (ci, wx), (cj, wy), (ck, wz) in corners:
            np.add.at(density_grid, (ci, cj, ck), wx*wy*wz) # mass weighted by distance of the particle to center of the cell
        
        return density_grid

    def window(self):
        """Fourier-space window of the CIC assignment scheme
        W(k) = [sin(k_x*l/2)/(k_x*l/2) * sin(k_y*l/2)/(k_y*l/2) * sin(k_z*l/2)/(k_z*l/2)]^2 = [sinc(k_x*l/2) * sinc(k_y*l/2) * sinc(k_z*l/2)]^2
        P(k) ~ P_measured/W^2 """
        f = np.fft.fftfreq(self.N)  # k_i*l/2 in units of pi
        s = np.sinc(f)  # sinc(k_i*l/2) = sin(k_i*l/2)/(k_i*l/2)
        return (s[:, None, None] * s[None, :, None] * s[None, None, :])**2

    def Pk_measured(self, nbins: int = 20, sample: str = "particles", kmax_frac: float = 0.5):
        """P_hat(k) = 1/(N_k V) sum_{k_a in S_k} |delta_k|^2 in nbins linear shells up to kmax_frac * k_Nyq
        for safe window deconvolution in particle samples (avoid aliasing).
        Returns (k, P(k), N(k)), where k is the mean k of the modes in each shell, P(k) the mean power and N(k) the number of modes"""
        #k-bins
        *_, k_mag = kgrid(self.N, self.L)
        k_f, k_nyq = 2*np.pi / self.L, np.pi * self.N / self.L
        bins = np.linspace(k_f, kmax_frac * k_nyq, nbins + 1)

        delta_k = self.delta_k_gauss if sample == "gaussian" else self.delta_k_part
        power = np.abs(delta_k)**2 / self.L**3  # |delta_k|^2 / V in each cell
        if sample != "gaussian":
            power = power / self.W**2  # deconvolution of the CIC window function in Fourier space

        k = k_mag.ravel()
        k_mean = binned_statistic(k, k, statistic='mean', bins=bins)[0] # mean k of the modes in each shell
        Pk = binned_statistic(k, power.ravel(), statistic='mean', bins=bins)[0] # mean power in each shell
        nk = binned_statistic(k, power.ravel(), statistic='count', bins=bins)[0] # number of modes in each shell
        return k_mean, Pk, nk