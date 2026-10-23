from __future__ import annotations

import numpy as np
import scipy.integrate

class Cosmology:
    def __init__(self, Omega_m0: float, Omega_lambda0: float):
        self.Omega_m0 = Omega_m0
        self.Omega_lambda0 = Omega_lambda0
        self.Omega_k0 = 1.0 - Omega_m0 - Omega_lambda0

    def E(self, a: float) -> float:
        return np.sqrt(self.Omega_m0*a**-3 + self.Omega_k0*a**-2 + self.Omega_lambda0)

    def omega_m(self, a: float) -> float:
        return self.Omega_m0 * a**-3 / self.E(a)**2

    def dlnH_dN(self, a):
        """dlnH/dN = dlnE/dN"""
        return  -(3.0*self.Omega_m0*a**-3 + 2.0*self.Omega_k0*a**-2) / (2.0*self.E(a)**2)

    def growing_mode(self, a, a_i: float = 1e-3, rtol: float = 1e-10, atol: float = 1e-14):
        """Returns (D(a), f(a)) from the ODE 
        G_NN + (2 + dlnH/dN) G_N - 3/2 Omega_m(a) G = 0, with N = ln a"""

        def ode(N, y):
            G, G_N = y
            a = np.exp(N)
            G_NN = 3/2*self.omega_m(a)*G - (2 + self.dlnH_dN(a))*G_N
            return [G_N, G_NN]
        
        a = np.atleast_1d(np.asarray(a, dtype=float))
        N_i, N_f = np.log(a_i), np.log(max(a.max(), 1))

        y0 = [a_i, a_i]  # initial conditions: G(a_i) = a_i, G_N(a_i) = a_i
        sol = scipy.integrate.solve_ivp(ode, [N_i, N_f], y0, rtol=rtol, atol=atol, method='DOP853', dense_output=True)
        G, G_N = sol.sol(np.log(a)) #2xn array where n is the length of a with G in row 0 and G_N in row 1
        G1 = sol.sol(0.0)[0]  # normalization G(a=1), (N(1) = 0)
        D, f = G/G1, G_N/G
        return D, f

    def growing_mode_quadrature(self, a):
        """G(a) = (5 Omega_m0 / 2) E(a) * int_0^a dã / (ã^3 E^3(ã))"""
        a = np.atleast_1d(np.asarray(a, dtype=float))
        integrand = lambda x: 1.0 / (x**3 * self.E(x)**3)
        I = np.array([scipy.integrate.quad(integrand, 0.0, ai, epsabs=0, epsrel=1e-12)[0] for ai in a])
        G  = 2.5*self.Omega_m0*self.E(a)*I
        f  = self.dlnH_dN(a) + 1/(a**2 * I * self.E(a)**3)
        G1 = 2.5*self.Omega_m0*self.E(1.0)*scipy.integrate.quad(integrand, 0.0, 1.0, epsabs=0, epsrel=1e-12)[0]
        return G / G1, f

