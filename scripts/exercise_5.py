from __future__ import annotations

from pathlib import Path

import numpy as np
import camb
import classy
from classy import Class
from scipy.interpolate import CubicSpline

EX5 = Path(__file__).resolve().parents[1] / 'ex5_class'
INI = EX5 / 'class_base.ini'          # working copy 
OUTPUT = EX5 / 'output'               # output directory
FILE_KEYS = ('format', 'headers', 'overwrite_root', 'root')   # only control text files of the CLASS binary
T_NU_STANDARD = (4/11)**(1/3)         # standard neutrino-to-photon temperature ratio

# ------------------------------ precision controls ------------------------------
# {category: {CLASS parameter: (default value, tight reference value)}}
# defaults read from include/precisions.h of CLASS v3.4.1 (l_max_scalars from class_base.ini)
CONTROLS = {
    'ODE tolerance':          {'tol_perturbations_integration': (1e-5, 1e-7)},
    'k sampling':             {'k_per_decade_for_pk': (10, 40),
                               'k_per_decade_for_bao': (70, 200),
                               'k_step_sub': (0.05, 0.015)},
    'time sampling':          {'perturbations_sampling_stepsize': (0.1, 0.02)},
    'hierarchy depth':        {'l_max_g': (12, 40),
                               'l_max_pol_g': (10, 30),
                               'l_max_ur': (17, 50),
                               'l_max_ncdm': (17, 50)},
    'approximation switches': {'tight_coupling_trigger_tau_c_over_tau_h': (0.015, 0.005),
                               'tight_coupling_trigger_tau_c_over_tau_k': (0.01, 0.003),
                               'radiation_streaming_trigger_tau_over_tau_k': (45, 120),
                               'ur_fluid_trigger_tau_over_tau_k': (30, 100),
                               'ncdm_fluid_trigger_tau_over_tau_k': (31, 101)},  # CLASS forbids equal ur/ncdm triggers
    'maximum multipole':      {'l_max_scalars': (3000, 5000)},
    'ncdm momentum sampling': {'tol_ncdm_synchronous': (1e-3, 1e-5),
                               'tol_ncdm_bg': (1e-5, 1e-7)},
}
REFERENCE = {name: ref for group in CONTROLS.values() for name, (_, ref) in group.items()}  # all controls tight


# ------------------------------ CLASS runs ------------------------------
def read_ini(path=INI):
    """Reads 'key = value' lines of a CLASS .ini file into a dict of strings (comments and file keys skipped)"""
    params = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#')[0].strip()       # removes comments
        if '=' not in line:
            continue
        key, value = (s.strip() for s in line.split('=', 1))
        if key not in FILE_KEYS:                 # classy returns arrays, it does not write files
            params[key] = value
    return params

def run_class(params, extra=None):
    """Runs CLASS (classy) with the .ini parameters plus optional extra settings, returns the computed instance"""
    cosmo = Class()
    cosmo.set({**params, **(extra or {})})
    cosmo.compute()
    return cosmo

def pk_lin(cosmo, k, z=0.0):
    """Solver's direct linear total-matter P_m(k, z) [Mpc^3] for k [1/Mpc]"""
    return np.array([cosmo.pk_lin(ki, z) for ki in k])

def record(cosmo, params, z=0.0):
    """Saves the code version, physical parameters, output redshift, gauge, matter component and units of a CLASS run"""
    derived = cosmo.get_current_derived_parameters(['Omega_m', 'Omega_Lambda', 'Neff', 'YHe', 'sigma8'])
    return {'code': f'CLASS {classy.__version__} (classy)', **params,
            'T_cmb [K]': cosmo.T_cmb(), 'Omega_ncdm': cosmo.Omega_nu, **derived,
            'z (output)': z,
            'gauge': 'synchronous (CLASS default)',
            'recombination': 'HyRec (CLASS default)',
            'matter component': 'total matter d_m (cdm + b + ncdm)',
            'units': 'transfer k in h/Mpc, d_m dimensionless (R = 1); pk_lin: k in 1/Mpc, P in Mpc^3'}

def reconstruct(cosmo, params, z=0.0):
    """P_m^rec = 2 pi^2/k^3 A_s (k/k_*)^(n_s - 1) T_m^2 on the CLASS transfer k grid, with T_m = d_m (= delta_m/R).
    Returns k [1/Mpc], T_m, P_rec and the direct P_m [Mpc^3] at the same k"""
    h, A_s, n_s, k_pivot = (float(params[key]) for key in ('h', 'A_s', 'n_s', 'k_pivot'))
    tk = cosmo.get_transfer(z, output_format='class')   # density transfers normalized to R = 1
    k = h*tk['k (h/Mpc)']  # k = h k_h  [1/Mpc]
    T_m = tk['d_m']   # total-matter density transfer
    P_rec = 2*np.pi**2/k**3 * A_s*(k/k_pivot)**(n_s - 1) * T_m**2
    return k, T_m, P_rec, pk_lin(cosmo, k, z)


# ------------------------------ convergence ------------------------------
def precision_ratios(params, k, z=0.0):
    """P_m ratios to the tight reference.
    'fiducial': all controls at default; each control name: reference with only that control back at default.
    Returns {'reference': P_ref, 'fiducial': P/P_ref, control: P/P_ref}"""
    P_ref = pk_lin(run_class(params, REFERENCE), k, z)
    out = {'reference': P_ref, 'fiducial': pk_lin(run_class(params), k, z)/P_ref}
    for group in CONTROLS.values():
        for name, (default, _) in group.items():
            cosmo = run_class(params, {**REFERENCE, name: default})   # one control relaxed at a time
            out[name] = pk_lin(cosmo, k, z)/P_ref
    return out


# ------------------------------ CAMB with matched physics ------------------------------
def run_camb(cosmo, params, high=False, z=0.0):
    """CAMB run with the physical parameters of the CLASS run cosmo:
    same h, omega_b, omega_cdm, Omega_k, A_s, n_s, k_pivot, tau_reio, T_cmb and YHe (CLASS BBN value);
    neutrinos: N_ur massless + 1 massive species with CLASS degeneracy (T_ncdm/T_nu)^4 and the same Omega_ncdm h^2;
    Recfast recombination and tanh reionization (as CLASS 'RECFAST' and 'reio_camb', same helium width);
    synchronous-gauge total matter (delta_tot), linear only.
    high = True: AccuracyBoost = 3, high_precision transfer and accurate massive-neutrino transfers"""
    h = float(params['h'])
    g_ncdm = (float(params['T_ncdm'])/T_NU_STANDARD)**4     # massive-species degeneracy (N_eff contribution)
    pars = camb.set_params(H0=100*h, ombh2=float(params['omega_b']), omch2=float(params['omega_cdm']),
                           omk=float(params['Omega_k']), mnu=float(params['m_ncdm']), num_massive_neutrinos=1,
                           nnu=float(params['N_ur']) + g_ncdm, TCMB=cosmo.T_cmb(),
                           YHe=cosmo.get_current_derived_parameters(['YHe'])['YHe'], tau=float(params['tau_reio']),
                           As=float(params['A_s']), ns=float(params['n_s']), pivot_scalar=float(params['k_pivot']),
                           WantCls=False)
    pars.num_nu_massless = float(params['N_ur'])     # same split as CLASS: N_ur + one massive species
    pars.nu_mass_degeneracies = [g_ncdm]
    pars.omnuh2 = cosmo.Omega_nu*h**2                # same massive-neutrino density as CLASS
    pars.Reion.helium_delta_redshift = 0.5           # CLASS reio_camb helium width (CAMB default 0.4)
    pars.set_matter_power(redshifts=[z], kmax=1.2*h*float(params['P_k_max_h/Mpc']), nonlinear=False,
                          accurate_massive_neutrino_transfers=high)
    if high:
        pars.set_accuracy(AccuracyBoost=3)
        pars.Transfer.high_precision = True
    return camb.get_results(pars)

def camb_pk(results, k, z=0.0):
    """CAMB linear total-matter P_m(k, z) [Mpc^3] for k [1/Mpc]"""
    PK = results.get_matter_power_interpolator(nonlinear=False, hubble_units=False, k_hunit=False,
                                               var1='delta_tot', var2='delta_tot')
    return PK.P(z, k)

def camb_transfer(results, h):
    """CAMB total-matter transfer: CAMB stores delta/k^2 (k in 1/Mpc, synchronous gauge, unit primordial curvature),
    so T_m = Transfer_tot k^2. Returns k [1/Mpc], T_m"""
    tr = results.get_matter_transfer_data()
    k = h*tr.transfer_data[camb.model.Transfer_kh - 1, :, 0]        # k/h -> k [1/Mpc]
    return k, tr.transfer_data[camb.model.Transfer_tot - 1, :, 0]*k**2

def class_transfer_at(cosmo, h, k, z=0.0):
    """CLASS T_m = d_m interpolated to k [1/Mpc] with a cubic spline in (ln k, ln|T_m|), sign kept"""
    tk = cosmo.get_transfer(z, output_format='class')
    spline = CubicSpline(np.log(h*tk['k (h/Mpc)']), np.log(np.abs(tk['d_m'])))
    return np.sign(tk['d_m'][0])*np.exp(spline(np.log(k)))
