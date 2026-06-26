import numpy as np, camb
from scipy import constants as const
from scipy.special import spherical_jn, gammaln
from scipy.interpolate import RectBivariateSpline, interp1d
from sympy.physics.wigner import wigner_3j
from functools import lru_cache

def normalization(n=-2.9):

    R_g = 1/(1+3.044*(7/8)*(4/11)**(4/3))
    rho_g = (np.pi**2/15)*(2.7255 * const.Boltzmann / const.electron_volt)**4 # eV^4
    GHz = 1e9*const.hbar/const.e # Hz -> m -> eV
    
    # Gaussian CGS
    q = np.sqrt(const.alpha)
    nG = 6.93e-11 # eV^2

    log_value = gammaln((n + 3) / 2)
    fac = np.exp(log_value)
    
    A_T = (-1/rho_g)*1.5*R_g*np.log(10**17)  # eV^{-4}
    A_a = 3./(32.*np.pi**4*(100*GHz)**2*q) # eV^{-2}
    A_B = 4*np.pi**2*nG**2/fac # eV^4
    
    norm = {}
    norm['Taa'] = A_T*A_a**2*A_B**2
    norm['Baa'] = -4.*np.sqrt(2.)*norm['Taa']
    norm['aa']  = 4*(np.pi)**3*A_a**2*A_B

    return norm


def compute_transfer_B():
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.WantScalars = True
    pars.WantTensors = True
    pars.set_accuracy(AccuracyBoost=2)
    data = camb.get_transfer_functions(pars)
    transfer = data.get_cmb_transfer_data('tensor')
    ells, ks, TB_lk = transfer.get_transfer(source=2)
    TB_lk /= 4.
    return ells, ks, TB_lk


def claa(lmax, eta0, nB, xn=100, lnxmin=-1, lnxmax=4):

    l_list = range(2, lmax)
    x = np.logspace(lnxmin,lnxmax,xn)
    xi_table = {}
    
    for ell in l_list:
        l = int(ell)        
        integrand = spherical_jn(l,x)**2 * x**nB
        xi_table[(l, 'aa')] = l*(l+1)*np.trapezoid(integrand, x)/eta0**(nB+3)
            
    return xi_table


def Xi(lmax, r, eta0, nB, kn=100, lnkmin=-5, lnkmax=-1.5, check_claa=False):
    """
    Precompute Xi_l^{spin, X}(r) on the supplied r grid.

    Returns
    -------
    xi_table : dict
        xi_table[(ell, spin)] = array over r
    """
    l_list = range(2, lmax)
    k = np.logspace(lnkmin,lnkmax,kn)
    x = np.outer(r, k)
    xi_table = {}
    
    for ell in l_list:
        
        l = int(ell)
        jl = spherical_jn(l, x)
        jlp = spherical_jn(l, x, derivative=True)
        y = k * eta0
        
        Talpha = np.sqrt(l*(l+1)) * spherical_jn(l, y) / y
        base = k**2 * Talpha * k**nB  # P(k)=k^nB

        if check_claa:
            integrand = base * Talpha
            xi_table[(l, 'aa')] = np.trapezoid(integrand, k)
        else:
            kernel = { 'B': jl, 'E': jl/x + jlp, 'L': np.sqrt(l*(l+1))*jl/x }
            for mode in ('B', 'E', 'L'):
                integrand = kernel[mode] * base[None, :]
                xi_table[(l, mode)] = np.trapezoid(integrand, k, axis=1)
            
    return xi_table


def Xi_func(lmax, r, eta0, nB, kn=100, kind="cubic"):
    Xi_table  = Xi(lmax, r, eta0, nB, kn=kn, check_claa=False)
    Xi_interp = {}
    for key, values in Xi_table.items():
        Xi_interp[key] = interp1d(r, values, kind=kind, bounds_error=True)
    return Xi_interp


def Upsilon_TS(lmax, r, T_theta, kn=100, lnkmax=-1.5):
    r"""
    Precompute Upsilon_l(r) for temperature scalar on the supplied r grid.

    Upsilon^{D,L,R}_l(r) = int dk K^{D,L,R} \Delta_l^{Theta,S}(k)

    Returns
    -------
    upsilon_table : dict
        upsilon_table[ell] = array over r.
    """
    l_list = range(2, lmax)
    k = np.logspace(-5,lnkmax,kn)
    x = np.outer(r, k)
    upsilon_table = {}
    for ell in l_list:
        l = int(ell)
        jl = spherical_jn(l, x)
        jlp = spherical_jn(l, x, derivative=True)
        Ttheta = T_theta(l,k)
        for mode in ['D','L','R']:
            if mode == 'D':
                kernel = ( x * jlp + x**2 * jl / 3.0 )
            elif mode == 'L':
                kernel = jl
            elif mode == 'R':
                kernel = ( x * jlp - jl )
            upsilon_table[(l,mode)] = np.trapezoid( kernel * Ttheta[None, :], k, axis=1 )
    return upsilon_table


def Upsilon_BT(lmax, r, Delta_BT, kn=100, lnkmax=-1.5):
    r"""
    Precompute \bar{Upsilon}^{TB/VB}_l(r) on the supplied r grid.

    Returns
    -------
    upsilon_table : dict
        upsilon_table[ell] = array over r.
    """
    l_list = range(2, lmax)
    k = np.logspace(-5,lnkmax,kn)
    x = np.outer(r, k)
    upsilon_table = {}
    for ell in l_list:
        l = int(ell)
        jl = spherical_jn(l, x)
        jlp = spherical_jn(l, x, derivative=True)
        Delta = Delta_BT(l,k)
        for mode in ['TB','VB']:
            if mode == 'TB':
                kernel = ( x**2*jlp + 2*x*jl )
            elif mode == 'VB':
                kernel = x*jl
            upsilon_table[(l,mode)] = np.trapezoid( kernel * Delta[None, :], k, axis=1 )
    return upsilon_table


def Upsilon_func(lmax, r, Transfer, perturb='TS', kn=100, kind="cubic"):

    if perturb == 'TS':
        Upsilon_table  = Upsilon_TS(lmax, r, Transfer, kn=kn)
    elif perturb == 'BT':
        Upsilon_table  = Upsilon_BT(lmax, r, Transfer, kn=kn)
    
    Upsilon_interp = {} 
    for key, values in Upsilon_table.items():
        Upsilon_interp[key] = interp1d(r, values, kind=kind, bounds_error=True)
    
    return Upsilon_interp


def parity(l,l1,l2,p):
    """
    Parity projector:
        p^p_{l l1 l2} = [1+p(-1)^{l+l1+l2}]/2
    """
    return 0.5*(1.0+p*(-1)**(l+l1+l2))

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


#////////////////////////////////////////////////////////////////////////////////////////////////////#
# Bispectrum for Taa
#////////////////////////////////////////////////////////////////////////////////////////////////////#

def integ_term_D(l, l1, l2, r, Upsilon, Xi):
    integrand = Upsilon[(l,'D')](r) * ( Xi[(l1,'E')](r) * Xi[(l2,'B')](r) - Xi[(l2,'E')](r) * Xi[(l1,'B')](r) )
    return 2.0 * Wigner(l,l1,l2,0,1,-1) * np.trapezoid(integrand, r)


def integ_term_L(l, l1, l2, r, Upsilon, Xi):
    W211 = Wigner(l,l1,l2,-2,1,1)
    W011 = Wigner(l,l1,l2,0,1,-1)
    pref12 = sqrt_factorial_ratio(l+2,l-2) * W211 + l*(l+1)*W011
    pref21 = sqrt_factorial_ratio(l+2,l-2) * (-W211) + l*(l+1)*W011
    integrand12 =  Upsilon[(l,'L')](r) * Xi[(l1,'E')](r) * Xi[(l2,'B')](r)
    integrand21 =  Upsilon[(l,'L')](r) * Xi[(l2,'E')](r) * Xi[(l1,'B')](r)
    return pref12 * np.trapezoid(integrand12, r) + pref12 * np.trapezoid(integrand21, r)

    
def integ_term_R(l, l1, l2, r, Upsilon, Xi):
    pref12 = 2.0 * sqrt_factorial_ratio(l+1,l-1) * Wigner(l,l1,l2,-1,0,1)
    pref21 = 2.0 * sqrt_factorial_ratio(l+1,l-1) * Wigner(l,l2,l1,-1,0,1)
    integrand12 = Upsilon[(l,'R')](r) * Xi[(l1,'L')](r) * Xi[(l2,'B')](r)
    integrand21 = Upsilon[(l,'R')](r) * Xi[(l2,'L')](r) * Xi[(l1,'B')](r)
    return - pref12 * np.trapezoid(integrand12, r) - pref21 * np.trapezoid(integrand21, r)


def bispec_Taa_odd(l,l1,l2,Upsilon,Xi,logrmin=2,logrmax=5,rn=100):
    r"""
    Compute the normalized reduced bispectrum, B^BH_{l l1 l2}/(A_T A^2_a A_B A_H)
    """
    r = np.logspace(logrmin,logrmax,rn)
    bispec_D = integ_term_D(l,l1,l2,r,Upsilon,Xi)
    bispec_L = integ_term_L(l,l1,l2,r,Upsilon,Xi)
    bispec_R = integ_term_R(l,l1,l2,r,Upsilon,Xi)
    return 2*1j * parity(l,l1,l2,-1) * gamma(l,l1,l2) * (bispec_D+bispec_L+bispec_R)



#////////////////////////////////////////////////////////////////////////////////////////////////////#
# Bispectrum for Baa
#////////////////////////////////////////////////////////////////////////////////////////////////////#

def integ_term_even(l, l1, l2, r, Upsilon, Xi_B, Xi_H):
    W211 = Wigner(l,l1,l2,2,-1,-1)
    W101 = Wigner(l,l1,l2,1,0,-1)
    integ1 = Upsilon[(l,'TB')](r) * Xi_B[(l1,'E')](r) * Xi_H[(l2,'B')](r) + Upsilon[(l,'TB')](r) * Xi_B[(l2,'E')](r) * Xi_H[(l1,'B')](r)
    integ2 = Upsilon[(l,'VB')](r) * Xi_B[(l1,'L')](r) * Xi_H[(l2,'B')](r) + Upsilon[(l,'VB')](r) * Xi_B[(l2,'L')](r) * Xi_H[(l1,'B')](r)
    return - W211 * np.trapezoid(integ1, r) + np.sqrt((l-1)*(l+2)) * W101 * np.trapezoid(integ2, r)


def integ_term_odd(l, l1, l2, r, Upsilon, Xi_B, Xi_H):
    W211 = Wigner(l,l1,l2,2,-1,-1)
    integrand1 =  Upsilon[(l,'TB')](r) * ( Xi_H[(l1,'B')](r) * Xi_H[(l2,'B')](r) - Xi_B[(l1,'E')](r) * Xi_B[(l2,'E')](r) )
    integrand2 =  Upsilon[(l,'VB')](r) * Xi_B[(l1,'L')](r) * Xi_B[(l2,'E')](r) 
    integrand3 =  Upsilon[(l,'VB')](r) * Xi_B[(l2,'L')](r) * Xi_B[(l1,'E')](r)
    term_TB = W211 * np.trapezoid(integrand1, r)
    term_VB = Wigner(l,l1,l2,1,0,-1) * np.trapezoid(integrand2, r) + Wigner(l,l2,l1,1,0,-1) * np.trapezoid(integrand3, r)
    return term_TB + np.sqrt((l-1)*(l+2))*term_VB

    
def bispec_Baa_even(l,l1,l2,Upsilon,Xi_B,Xi_H,logrmin=2,logrmax=5,rn=100):
    r"""
    Compute the normalized reduced bispectrum, B^even_{l l1 l2}/(A_T A^2_a A_B A_H)
    """
    r = np.logspace(logrmin,logrmax,rn)
    bispec_even = integ_term_even(l,l1,l2,r,Upsilon,Xi_B,Xi_H)
    return 2 * parity(l,l1,l2,1) * gamma(l,l1,l2) * bispec_even


def bispec_Baa_odd(l,l1,l2,Upsilon,Xi_B,Xi_H,logrmin=2,logrmax=5,rn=100):
    r"""
    Compute the normalized reduced bispectrum, B^odd_{l l1 l2}/(A_T A^2_a A_B A_H)
    """
    r = np.logspace(logrmin,logrmax,rn)
    bispec_odd = integ_term_odd(l,l1,l2,r,Upsilon,Xi_B,Xi_H)
    return -2*1j * parity(l,l1,l2,-1) * gamma(l,l1,l2) * bispec_odd

