from __future__ import annotations

from abc import ABC, abstractmethod   

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from exercise_1 import Cosmology

DELTA_C_EDS = 3/20 * (12*np.pi)**(2/3)   # EdS linear collapse threshold 

# --------------------- virial prescription --------------------
def virial_y(lambda_ta: float):
    """returns the root of 4 lambda_ta y^3 - 2(1 + lambda_ta) y + 1 = 0 in (0, 1/2]
    that tends to 1/2 (EdS) as lambda_ta -> 0)"""
    if lambda_ta == 0.0:
        return 0.5      # EdS: y = 1/2
    cubic = lambda y: 4*lambda_ta*y**3 - 2*(1 + lambda_ta)*y + 1
    return brentq(cubic, 0.0, 0.5, xtol=1e-15, rtol=1e-15)  # finds root in (0, 1/2)

# --------------------- Background and formulations A, B, C --------------------
class Background(ABC):
    """Flat background (lambdaCDM or EdS without radiation) shared by all formulations with N = ln a"""
    collapse_threshold = None   # collapse condition used when none is given, set in subclasses A, B, C

    def __init__(self, Omega_m0: float, threshold: float | None = None, a_i: float = 1e-5,
                 rtol: float = 1e-10, atol: float = 1e-13):
        self.cosmo = Cosmology(Omega_m0, 1.0 - Omega_m0)
        self.threshold = self.collapse_threshold if threshold is None else threshold   # delta_inf (A, B) or eps (C)
        self.a_i = a_i
        self.rtol, self.atol = rtol, atol                               # solve_ivp tolerances
        self.f_i = self.cosmo.growing_mode(a_i, a_i=a_i*1e-3)[1][0]   # f = G_N/G at a_i, ODE starts at 0.001 a_i

    def background(self, N):
        """A_H, Omega_m for each N"""
        a = np.exp(N)
        return 2.0 + self.cosmo.dlnH_dN(a), self.cosmo.omega_m(a)

    @abstractmethod
    def rhs(self, N, y):
        """Nonlinear equations of provided by each formulation, defined in subclasses A, B, C
        y = 2 nonlinear variables"""
        ...

    def linear_nonlinear_rhs(self, N, y):
        """Returns the linear equation d^2(delta_L)/dN^2 = - A_H d(delta_L)/dN + 3/2 O_m delta_L
        written as a first-order system and the A, B or C formulation for integration on the same steps
        y = [y0, y1, delta_L, d(delta_L)/dN], where y0, y1 = delta, d(delta)/dN in A ; delta, theta in B ; g, theta in C"""
        AH, Om = self.background(N)
        dL, dL_N = y[2], y[3]
        return [*self.rhs(N, y[:2]), dL_N, -AH*dL_N + 1.5*Om*dL]

    @abstractmethod
    def initial_conditions(self, delta_i):
        """Initial conditions of the 2 nonlinear variables at a_i, defined in subclasses A, B, C"""
        ...

    @abstractmethod
    def collapse_epoch(self, N, y):
        """Determines when delta non linear reaches delta_inf (collapse threshold), defined in subclasses A, B, C. 
        Returns 0 when threshold is reached"""
        ...

    @abstractmethod
    def theta(self, y):
        """theta for three formulations, defined in subclasses A, B, C"""
        ...

    @abstractmethod
    def delta(self, y):
        """delta for three formulations, defined in subclasses A, B, C"""
        ...

    def omega_lambda(self, a):
        """Omega_Lambda(a) = Omega_Lambda0 / E^2(a)"""
        return self.cosmo.Omega_lambda0 / self.cosmo.E(a)**2

    def integrate(self, delta_i, N_end):
        """Integration of formulation (A, B or C) and linear equation from N_i = ln a_i to N_end,
        with initial linear conditions at a_i: delta_L,i = delta_i, d(delta_L)/dN,i = f_i delta_i
        and nonlinear given by initial_conditions(delta_i) of the formulation.
        Stops when delta non linear reaches collapse threshold.
        sol.t_events[0] = N at collapse (empty if no collapse before N_end)
        sol.t_events[1], sol.y_events[1] = N and state at turnaround, theta = -3 (dlnR/dN = 1 + theta/3 = 0)"""
        event = lambda N, y: self.collapse_epoch(N, y)
        event.terminal = True # setting to stop integration at threshold
        turnaround = lambda N, y: self.theta(y) + 3.0
        turnaround.direction = -1   # setting to record moment that theta = -3
        y0 = [*self.initial_conditions(delta_i), delta_i, self.f_i*delta_i]
        return solve_ivp(self.linear_nonlinear_rhs, [np.log(self.a_i), N_end], y0, method='DOP853',
                         rtol=self.rtol, atol=self.atol, events=[event, turnaround], dense_output=True)

    def tune_delta_i(self, z_c: float):
        """Tunes delta_i until so collapse event happens at a_c = 1/(1+z_c) and reads delta_c = delta_L(a_c).
        Residual r(delta_i) = N_coll - N_c (if > 0, collapse happened too late, if < 0, too early), 
        Integrates to N_c + 0.5 (N_coll = N_c + 0.5 if no collapse by then). 
        Bracket: [lo, hi] interval of trial δ_i = +-3% around linear guess
        Linear guess: 1.686 D(a_i)/D(a_c) widened until r changes sign; root is found with brentq.
        Returns a dict with tuned run, root-search record and settings"""
        N_c = -np.log(1 + z_c)
        N_end = N_c + 0.5   # value past a_c to look for late collapse

        def residual(delta_i):
            sol = self.integrate(delta_i, N_end)
            N_coll = sol.t_events[0][0] if sol.t_events[0].size else N_end
            return N_coll - N_c

        D = self.cosmo.growing_mode(np.array([self.a_i, np.exp(N_c)]), a_i=self.a_i*1e-3)[0]
        guess = DELTA_C_EDS * D[0]/D[1]  # linear guess for delta_i
        lo, hi = 0.97*guess, 1.03*guess
        while residual(lo) < 0: lo *= 0.97  # widens bracket until the sign changes
        while residual(hi) > 0: hi *= 1.03
        delta_i, info = brentq(residual, lo, hi, xtol=1e-14*guess, rtol=1e-14, full_output=True)

        sol = self.integrate(delta_i, N_end)  # tuned run
        return dict(formulation=type(self).__name__, Omega_m0=self.cosmo.Omega_m0, z_c=z_c,
                    delta_i=delta_i, delta_c=sol.sol(N_c)[2], # delta_L at a_c
                    bracket=(float(lo), float(hi)), iterations=info.iterations,
                    event_residual=sol.t_events[0][0] - N_c,  # ln(a_collapse / a_c)
                    nfev=sol.nfev,  # right-hand-side evaluations of the tuned run (cost)
                    a_i=self.a_i, threshold=self.threshold, rtol=self.rtol, atol=self.atol)

    def turnaround_virial(self, z_c: float):
        """Tuned run (tune_delta_i) + turnaround and virial overdensities.
        Turnaround: event theta = -3 on the tuned run, zeta_ta = 1 + delta(a_ta).
        Virial: lambda_ta = Omega_Lambda(a_ta) / (Omega_m(a_ta) zeta_ta), y from virial_y,
        Delta_vir,m = zeta_ta (a_c/a_ta)^3 y^-3, Delta_vir,c = Omega_m(a_c) Delta_vir,m.
        Returns tune_delta_i dict, a_ta, a_ta/a_c, zeta_ta, lambda_ta, y, Delta_vir,m, Delta_vir,c"""
        out = self.tune_delta_i(z_c)
        a_c = 1/(1 + z_c)
        sol = self.integrate(out['delta_i'], np.log(a_c) + 0.5) # same tuned run
        N_ta, y_ta = sol.t_events[1][0], sol.y_events[1][0]  # turnaround event
        a_ta = np.exp(N_ta)
        zeta_ta = 1 + self.delta(y_ta)  # 1 + delta at turnaround

        lambda_ta = self.omega_lambda(a_ta) / (self.cosmo.omega_m(a_ta)*zeta_ta)   # rho_Lambda / rho_sph at turnaround
        y = virial_y(lambda_ta)   # R_vir / R_ta
        Delta_vir_m = zeta_ta * (a_c/a_ta)**3 / y**3  # relative to mean matter density at a_c
        Delta_vir_c = self.cosmo.omega_m(a_c) * Delta_vir_m  # relative to critical density at a_c
        out.update(a_ta=a_ta, a_ta_over_a_c=a_ta/a_c, zeta_ta=zeta_ta, lambda_ta=lambda_ta, y=y,
                   Delta_vir_m=Delta_vir_m, Delta_vir_c=Delta_vir_c)
        return out

class A(Background):
    """A: d^2(delta)/dN^2 = - A_H d(delta)/dN + 4/3 (d(delta)/dN)^2 / (1 + delta) + 3/2 O_m delta (1 + delta)
    written as a first-order system y = [delta, d(delta)/dN]"""
    collapse_threshold = 1e8   # delta_inf

    def rhs(self, N, y):
        d, d_N = y
        AH, Om = self.background(N)
        d_NN = -AH*d_N + 4/3*d_N**2/(1 + d) + 1.5*Om*d*(1 + d)
        return [d_N, d_NN]

    def initial_conditions(self, delta_i):
        """delta(a_i) = delta_i,  d(delta)/dN (a_i) = f_i delta_i"""
        return [delta_i, self.f_i*delta_i]

    def collapse_epoch(self, N, y):
        """delta - delta_inf"""
        return y[0] - self.threshold

    def theta(self, y):
        """theta = -d(delta)/dN / (1 + delta)"""
        return -y[1]/(1 + y[0])

    def delta(self, y):
        return y[0]

class B(Background):
    """B: d(delta)/dN = - (1 + delta) theta, d(theta)/dN = -A_H theta - theta^2 / 3 - 3/2 O_m delta 
    where theta = div_x(vec(v)) / a H is the velocity divergence, y = [delta, theta]"""
    collapse_threshold = 1e8   # delta_inf

    def rhs(self, N, y):
        AH, Om = self.background(N)
        d, th = y
        d_N = -(1 + d)*th   
        th_N = -AH*th - th**2/3 - 1.5*Om*d
        return [d_N, th_N]

    def initial_conditions(self, delta_i):
        """delta(a_i) = delta_i, theta_i = -f_i delta_i / (1 + delta_i)"""
        return [delta_i, -self.f_i*delta_i/(1 + delta_i)]

    def collapse_epoch(self, N, y):
        """delta - delta_inf"""
        return y[0] - self.threshold

    def theta(self, y):
        return y[1]

    def delta(self, y):
        return y[0]

class C(Background):
    """C: dg/dN = g (1 + g) theta, d(theta)/dN = -A_H theta - theta^2 / 3 - 3 O_m / 2g
    where g = 1/delta, y = [g, theta]"""
    collapse_threshold = 1e-8   # for delta_inf = 1/g_eps

    def rhs(self, N, y):
        AH, Om = self.background(N)
        g, th = y
        g_N = g*(1 + g)*th                       
        th_N = -AH*th - th**2/3 - 1.5*Om/g
        return [g_N, th_N]

    def initial_conditions(self, delta_i):
        """g_i = 1/delta_i, theta_i = -f_i delta_i / (1 + delta_i)"""
        return [1/delta_i, -self.f_i*delta_i/(1 + delta_i)]

    def collapse_epoch(self, N, y):
        """g - eps"""
        return y[0] - self.threshold

    def theta(self, y):
        return y[1]

    def delta(self, y):
        """delta = 1/g"""
        return 1/y[0]

if __name__ == '__main__':
    # command-line use: python exercise_4.py C 0.3 0.0 [--threshold 1e-8 --a_i 1e-5 --rtol 1e-10 --atol 1e-13]
    import argparse
    p = argparse.ArgumentParser(description='Spherical collapse: tuned run, turnaround and virial outputs')
    p.add_argument('formulation', choices=['A', 'B', 'C'])
    p.add_argument('Omega_m0', type=float)                   # flat: Omega_Lambda0 = 1 - Omega_m0 (EdS: 1)
    p.add_argument('z_c', type=float)                        # target collapse redshift
    p.add_argument('--threshold', type=float, default=None)  # default: 1e8 (A, B), 1e-8 (C)
    p.add_argument('--a_i', type=float, default=1e-5)
    p.add_argument('--rtol', type=float, default=1e-10)
    p.add_argument('--atol', type=float, default=1e-13)
    args = p.parse_args()
    F = {'A': A, 'B': B, 'C': C}[args.formulation]
    out = F(args.Omega_m0, args.threshold, args.a_i, args.rtol, args.atol).turnaround_virial(args.z_c)
    for key, value in out.items():   # one required output per line
        print(f'{key:15s} {value}')
