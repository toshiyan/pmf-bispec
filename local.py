import numpy as np
from scipy.special import spherical_jn, gammaln
from scipy.interpolate import RectBivariateSpline, interp1d
from sympy.physics.wigner import wigner_3j
from functools import lru_cache


def Xi(lmax, r, eta0, nB, kn=100):
    """
    Precompute Xi_l^{spin, X}(r) on the supplied r grid.

    Returns
    -------
    xi_table : dict
        xi_table[(ell, spin)] = array over r
    """
    l_list = range(2, lmax)
    k = np.logspace(-5,-1,kn)
    x = np.outer(r, k)
    xi_table = {}
    for ell in l_list:
        l = int(ell)
        jl = spherical_jn(l, x)
        jlp = spherical_jn(l, x, derivative=True)
        y = k * eta0
        Talpha = (1j) ** (-l+1) * np.sqrt(l*(l+1)) * spherical_jn(l, y) / y
        base = k**2 * Talpha * k**nB  # P(k)=k^nB
        for spin in (0, +1, -1):
            if spin == 0:
                kernel = jl
            elif spin == +1:
                kernel = -1j * (jl / x + jlp)
            elif spin == -1:
                kernel = -1j * np.sqrt(l*(l+1)) * jl / x
            integrand = kernel * base[None, :]
            xi_table[(l, spin)] = 2 * np.trapezoid(integrand, k, axis=1)
            
    return xi_table


def Xi_func(lmax, r, eta0, nB, kn=100, kind="cubic"):
    Xi_table  = Xi(lmax, r, eta0, nB, kn=kn)
    Xi_interp = {}
    for key, values in Xi_table.items():
        Xi_interp[key] = interp1d(r, values, kind=kind, bounds_error=True)
    return Xi_interp


def Upsilon(lmax, r, T_theta, kn=100):
    """
    Precompute Upsilon^delta_l(r) on the supplied r grid.

    Upsilon^{D,L,R}_l(r) = int dk K^{D,L,R} T_l^Theta(k)

    Returns
    -------
    upsilon_table : dict
        upsilon_table[ell] = array over r.
    """
    l_list = range(2, lmax)
    k = np.logspace(-5,-1,kn)
    x = np.outer(r, k)
    upsilon_table = {}
    for ell in l_list:
        l = int(ell)
        jl = spherical_jn(l, x)
        jlp = spherical_jn(l, x, derivative=True)
        Ttheta = (-1j) ** l * T_theta(l,k)
        for ut in ['D','L','R']:
            if ut == 'D':
                kernel = ( x * jlp + x**2 * jl / 3.0 )
            elif ut == 'L':
                kernel = jl
            elif ut == 'R':
                kernel = ( x * jlp - jl )
            upsilon_table[(l,ut)] = 2 * np.trapezoid( kernel * Ttheta[None, :], k, axis=1 )
    return upsilon_table


def Upsilon_func(lmax, r, T_theta, kn=100, kind="cubic"):
    Upsilon_table  = Upsilon(lmax, r, T_theta, kn=kn)
    Upsilon_interp = {} 
    for key, values in Upsilon_table.items():
        Upsilon_interp[key] = interp1d(r, values, kind=kind, bounds_error=True)
    return Upsilon_interp


def p_minus(l,l1,l2):
    """
    Odd-parity projector:
        p^-_{l l1 l2} = [1-(-1)^{l+l1+l2}]/2
    """
    return 0.5*(1.0-(-1)**(l+l1+l2))

def gamma(l, l1, l2):
    """
    gamma_{l l1 l2}
    """
    return np.sqrt( (2*l+1)*(2*l1+1)*(2*l2+1)/(4.0*np.pi) )

@lru_cache(maxsize=None)
def Wigner(l, l1, l2, m, m1, m2):
    if m + m1 + m2 != 0:
        return 0.0

    if abs(m) > l or abs(m1) > l1 or abs(m2) > l2:
        return 0.0

    if l < 0 or l1 < 0 or l2 < 0:
        return 0.0

    if abs(l1 - l2) > l or l > l1 + l2:
        return 0.0

    try:
        return float(wigner_3j(l, l1, l2, m, m1, m2))
    except ValueError:
        return 0.0

def sqrt_factorial_ratio(a, b):
    """
    sqrt(a! / b!) using log-gamma.
    """
    return np.exp(0.5 * (gammaln(a + 1) - gammaln(b + 1)))

def integ_term_D(l, l1, l2, r, Upsilon, Xi):
    integrand = Upsilon[(l,'D')](r) * ( Xi[(l1,+1)](r) * Xi[(l2,0)](r) - Xi[(l2,+1)](r) * Xi[(l1,0)](r) )
    return -2.0 * Wigner(l,l1,l2,0,1,-1) * np.trapezoid(integrand, r)

def integ_term_L(l, l1, l2, r, Upsilon, Xi):
    W211 = Wigner(l,l1,l2,-2,1,1)
    W011 = Wigner(l,l1,l2,0,1,-1)
    pref12 = sqrt_factorial_ratio(l+2,l-2) * W211 + l*(l+1)*W011
    pref21 = sqrt_factorial_ratio(l+2,l-2) * (-W211) + l*(l+1)*W011
    integrand12 =  Upsilon[(l,'L')](r) * Xi[(l1,+1)](r) * Xi[(l2,0)](r)
    integrand21 =  Upsilon[(l,'L')](r) * Xi[(l2,+1)](r) * Xi[(l1,0)](r)
    return - pref12 * np.trapezoid(integrand12, r) + pref12 * np.trapezoid(integrand21, r)

def integ_term_R(l, l1, l2, r, Upsilon, Xi):
    pref12 = 2.0 * sqrt_factorial_ratio(l+1,l-1) * Wigner(l,l1,l2,-1,0,1)
    pref21 = 2.0 * sqrt_factorial_ratio(l+1,l-1) * Wigner(l,l2,l1,-1,0,1)
    integrand12 = Upsilon[(l,'R')](r) * Xi[(l1,-1)](r) * Xi[(l2,0)](r)
    integrand21 = Upsilon[(l,'R')](r) * Xi[(l2,-1)](r) * Xi[(l1,0)](r)
    return pref12 * np.trapezoid(integrand12, r) - pref21 * np.trapezoid(integrand21, r)


def bispec(l,l1,l2,Upsilon,Xi,logrmin=2,logrmax=5,rn=100):
    """
    Compute B_{l l1 l2}
    """
    r = np.logspace(logrmin,logrmax,rn)
    bispec_D = integ_term_D(l,l1,l2,r,Upsilon,Xi)
    bispec_L = integ_term_L(l,l1,l2,r,Upsilon,Xi)
    bispec_R = integ_term_R(l,l1,l2,r,Upsilon,Xi)
    return (-1j)**(l+l1+l2) * p_minus(l,l1,l2) * gamma(l,l1,l2) * (bispec_D+bispec_L+bispec_R)

    