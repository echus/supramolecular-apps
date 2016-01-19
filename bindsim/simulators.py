from __future__ import division
from __future__ import print_function

import numpy as np
from math import sqrt

# TODO make cubic setup and solver a generalised class, with separate functions
# only dealing with differing molefrac/isotherm calculations

# Constants
N = 100 # Number of steps in simulation

def nmr_1to1(k1=1000,
            h0_init=0.001,
            g0h0_init=0,
            g0h0_final=20,
            dh=8,
            dhg=9,
            num=N):
    """
    NMR 1:1 binding constant simulator

    Args:
        k1        : float  Binding constant Ka
        h0_init   : float  Initial [H]0
        g0h0_init : float  Min equiv. [G]0/[H]0
        g0h0_final: float  Max equiv. [G]0/[H]0
        dh        : float  Free host NMR resonance
        dhg       : float  Host-Guest complex NMR resonance
        num       : int    Number of steps to evaluate at

    Returns:
        (g0h0,     : tuple, length 4 of arrays of length num
         dd,         Simulated isotherm and molefraction curves
         mf_h,
         mf_hg)

    Raises:
        [none]
    """

    # Initialise x array
    g0h0 = np.linspace(g0h0_init, g0h0_final, num)

    # Initialise y arrays
    s = g0h0.shape
    dd = np.zeros(s)       # Change in chemical shift delta (ppm or Hz)
    mf_h = np.zeros(s)         # H free host molefraction
    mf_hg = np.zeros(s)      # HG complex molefraction

    for i, geq in enumerate(g0h0):
        # For convenience
        h0 = h0_init
        g0 = geq * h0
        ka = k1

        dh = dh
        dhg = dhg

        # Calculate host and complex molefractions
        cfrac = (h0 + (1/ka) + g0) - sqrt((h0+(1/ka)+g0)**2 - 4*h0*g0)
        cfrac *= 0.5
        cfrac /= h0

        hfrac = 1 - cfrac

        # Calculate delta shift
        delta = dh + ((dhg - dh)*cfrac)

        dd[i] = delta
        mf_h[i] = hfrac
        mf_hg[i] = cfrac

    return g0h0, dd, mf_h, mf_hg

def uv_1to1(k1=1000,
            h0_init=0.001,
            g0h0_init=0,
            g0h0_final=20,
            dh=8,
            dhg=9,
            num=N):
    """
    UV 1:1 binding constant simulator

    Args:
        k1        : float  Binding constant Ka
        h0_init   : float  Initial [H]0
        g0h0_init : float  Min equiv. [G]0/[H]0
        g0h0_final: float  Max equiv. [G]0/[H]0
        dh        : float  Free host NMR resonance
        dhg       : float  Host-Guest complex NMR resonance
        num       : int    Number of steps to evaluate at

    Returns:
        (g0h0,     : tuple, length 4 of arrays of length num
         dd,         Simulated isotherm and molefraction curves
         mf_h,
         mf_hg)

    Raises:
        [none]
    """

    # Initialise x array
    g0h0 = np.linspace(g0h0_init, g0h0_final, num)

    # Initialise y arrays
    s = g0h0.shape
    dd = np.zeros(s)       # Change in chemical shift delta (ppm or Hz)
    mf_h = np.zeros(s)     # H free host molefraction
    mf_hg = np.zeros(s)    # HG complex molefraction

    for i, geq in enumerate(g0h0):
        # For convenience
        h0 = h0_init
        g0 = geq * h0
        ka = k1

        dh = dh
        dhg = dhg

        # Calculate host and complex molefractions
        cfrac = (h0 + (1/ka) + g0) - sqrt((h0+(1/ka)+g0)**2 - 4*h0*g0)
        cfrac *= 0.5
        cconc = cfrac # Save raw concentration for use in calculating delta
        cfrac /= h0

        hfrac = 1 - cfrac

        # Calculate delta shift
        delta = dh*(h0 - cconc) + dhg*cconc

        dd[i] = delta
        mf_h[i] = hfrac
        mf_hg[i] = cfrac

    return g0h0, dd, mf_h, mf_hg

def nmr_1to2(k1=10000,
             k2=1000,
             h0_init=0.001,
             g0h0_init=0,
             g0h0_final=20,
             dh=8,
             dhg=7,
             dhg2=9,
             num=N):
    """
    NMR 1:2 binding constant simulator

    Args:
        k1         : float  Binding constant K1
        k2         : float  Binding constant K2
        h0_init    : float  Initial [H]0
        g0h0_init  : float  Min equiv. [G]0/[H]0
        g0h0_final : float  Max equiv. [G]0/[H]0
        dh         : float  Free host NMR resonance
        dhg        : float  Host-Guest complex NMR resonance
        dhg2       : float  Host-Guest2 complex NMR resonance
        num        : int    Number of steps to evaluate at

    Returns:
        (g0h0,     : tuple, length 5 of arrays of length num
         dd,         Simulated isotherm and molefraction curves
         mf_h,
         mf_hg,
         mf_hg2)

    Raises:
        [none]
    """

    # Convert all input to float64s
    k1         = np.float64(k1)
    k2         = np.float64(k2)
    h0_init    = np.float64(h0_init)
    g0h0_init  = np.float64(g0h0_init)
    g0h0_final = np.float64(g0h0_final)
    dh         = np.float64(dh)
    dhg        = np.float64(dhg)
    dhg2       = np.float64(dhg2)

    dtype = np.dtype('f8')

    # Initialise x array
    g0h0 = np.linspace(g0h0_init, g0h0_final, num, dtype=dtype)

    # Initialise y arrays
    shape = g0h0.shape
    dd = np.zeros(shape, dtype)       # Change in chemical shift delta (ppm or Hz)
    mf_h = np.zeros(shape, dtype)     # H free host molefraction
    mf_hg = np.zeros(shape, dtype)    # HG complex molefraction
    mf_hg2 = np.zeros(shape, dtype)   # HG2 complex molefraction

    for i, geq in enumerate(g0h0):
        # For convenience
        h0 = h0_init
        g0 = geq * h0

        # Calculate reduced cubic coefficients (for cubic in [G])
        a = 1
        b = 2*h0 - g0 + 1/k2;
        c = h0/k2 - g0/k2 + 1/(k1*k2);
        d = (-g0)/(k1*k2);

        p = np.array([a, b, c, d], dtype=dtype)

        # Find cubic roots (solve for [G])
        roots = np.roots(p).astype(dtype)

        # Find smallest real positive root:
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        g = roots[select].min()
        g = float(np.real(g))

        mf_hg[i]  = (g*k1)/(1 + k1*g + k2*k1*(g**2))
        mf_hg2[i] = (k2*k1*(g**2))/(1 + k1*g + k2*k1*(g**2))
        mf_h[i]   = 1 - mf_hg[i] - mf_hg2[i]
        dd[i]     = dh*mf_h[i] + dhg*mf_hg[i] + dhg2*mf_hg2[i]

    return g0h0, dd, mf_h, mf_hg, mf_hg2

def uv_1to2(k1=10000,
            k2=1000,
            h0_init=0.001,
            g0h0_init=0,
            g0h0_final=20,
            dh=8,
            dhg=7,
            dhg2=9,
            num=N):
    """
    UV 1:2 binding constant simulator

    Args:
        k1         : float  Binding constant K1
        k2         : float  Binding constant K2
        h0_init    : float  Initial [H]0
        g0h0_init  : float  Min equiv. [G]0/[H]0
        g0h0_final : float  Max equiv. [G]0/[H]0
        dh         : float  Free host NMR resonance
        dhg        : float  Host-Guest complex NMR resonance
        dhg2       : float  Host-Guest2 complex NMR resonance
        num        : int    Number of steps to evaluate at

    Returns:
        (g0h0,     : tuple, length 5 of arrays of length num
         dd,         Simulated isotherm and molefraction curves
         mf_h,
         mf_hg,
         mf_hg2)

    Raises:
        [none]
    """

    # Convert all input to float64s
    k1         = np.float64(k1)
    k2         = np.float64(k2)
    h0_init    = np.float64(h0_init)
    g0h0_init  = np.float64(g0h0_init)
    g0h0_final = np.float64(g0h0_final)
    dh         = np.float64(dh)
    dhg        = np.float64(dhg)
    dhg2       = np.float64(dhg2)

    dtype = np.dtype('f8')

    # Initialise x array
    g0h0 = np.linspace(g0h0_init, g0h0_final, num, dtype=dtype)

    # Initialise y arrays
    shape = g0h0.shape
    dd = np.zeros(shape, dtype)       # Change in chemical shift delta (ppm or Hz)
    mf_h = np.zeros(shape, dtype)     # H free host molefraction
    mf_hg = np.zeros(shape, dtype)    # HG complex molefraction
    mf_hg2 = np.zeros(shape, dtype)   # HG2 complex molefraction

    for i, geq in enumerate(g0h0):
        # For convenience
        h0 = h0_init
        g0 = geq * h0

        # Calculate reduced cubic coefficients (for cubic in [G])
        a = 1
        b = 2*h0 - g0 + 1/k2;
        c = h0/k2 - g0/k2 + 1/(k1*k2);
        d = (-g0)/(k1*k2);

        p = np.array([a, b, c, d], dtype=dtype)

        # Find cubic roots (solve for [G])
        roots = np.roots(p).astype(dtype)

        # Find smallest real positive root:
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        g = roots[select].min()
        g = float(np.real(g))

        mf_hg[i]  = (g*k1)/(1 + k1*g + k2*k1*(g**2))
        mf_hg2[i] = (k2*k1*(g**2))/(1 + k1*g + k2*k1*(g**2))
        mf_h[i]   = 1 - mf_hg[i] - mf_hg2[i]
        dd[i]     = h0*(dh*mf_h[i] + dhg*mf_hg[i] + dhg2*mf_hg2[i])

    return g0h0, dd, mf_h, mf_hg, mf_hg2

def nmr_2to1(k1=10000,
             k2=1000,
             h0_init=0.001,
             g0h0_init=0,
             g0h0_final=20,
             dh=8,
             dhg=7,
             dh2g=9,
             num=N):
    """
    NMR 2:1 binding constant simulator

    Args:
        k1         : float  Binding constant K1
        k2         : float  Binding constant K2
        h0_init    : float  Initial [H]0
        g0h0_init  : float  Min equiv. [G]0/[H]0
        g0h0_final : float  Max equiv. [G]0/[H]0
        dh         : float  Free host NMR resonance
        dhg        : float  Host-Guest complex NMR resonance
        dh2g       : float  Host-Guest2 complex NMR resonance
        num        : int    Number of steps to evaluate at

    Returns:
        (g0h0,     : tuple, length 5 of arrays of length num
         dd,         Simulated isotherm and molefraction curves
         mf_h,
         mf_hg,
         mf_h2g)

    Raises:
        [none]
    """

    # Convert all input to float64s
    k1         = np.float64(k1)
    k2         = np.float64(k2)
    h0_init    = np.float64(h0_init)
    g0h0_init  = np.float64(g0h0_init)
    g0h0_final = np.float64(g0h0_final)
    dh         = np.float64(dh)
    dhg        = np.float64(dhg)
    dh2g       = np.float64(dh2g)

    dtype = np.dtype('f8')

    # Initialise x array
    g0h0 = np.linspace(g0h0_init, g0h0_final, num, dtype=dtype)

    # Initialise y arrays
    shape = g0h0.shape
    dd = np.zeros(shape, dtype)       # Change in chemical shift delta (ppm or Hz)
    mf_h = np.zeros(shape, dtype)     # H free host molefraction
    mf_hg = np.zeros(shape, dtype)    # HG complex molefraction
    mf_h2g = np.zeros(shape, dtype)   # H2G complex molefraction

    for i, geq in enumerate(g0h0):
        # For convenience
        h0 = h0_init
        g0 = geq * h0

        # Calculate reduced cubic coefficients (for cubic in [H])
        a = 1
        b = 2*g0 - h0 + 1/k2;
        c = g0/k2 - h0/k2 + 1/(k1*k2);
        d = (-h0)/(k1*k2);

        p = np.array([a, b, c, d], dtype=dtype)

        # Find cubic roots (solve for [H])
        roots = np.roots(p).astype(dtype)

        # Find smallest real positive root:
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        h = roots[select].min()
        h = float(np.real(h))

        mf_hg[i]  = (g0*h*k1)/(h0*(1 + (k1*h) + (k2*k1*h*h)))
        mf_h2g[i] = (2*g0*k2*k1*h*h)/(h0*(1 + (k1*h) + (k2*k1*h*h)))
        mf_h[i]   = 1 - mf_hg[i] - mf_h2g[i]
        dd[i]     = dh*mf_h[i] + dhg*mf_hg[i] + dh2g*mf_h2g[i]

    return g0h0, dd, mf_h, mf_hg, mf_h2g

def uv_2to1(k1=1000000,
            k2=100000,
            h0_init=0.00001,
            g0h0_init=0,
            g0h0_final=20,
            dh=20000,
            dhg=10000,
            dh2g=50000,
            num=N):
    """
    UV 2:1 binding constant simulator

    Args:
        k1         : float  Binding constant K1
        k2         : float  Binding constant K2
        h0_init    : float  Initial [H]0
        g0h0_init  : float  Min equiv. [G]0/[H]0
        g0h0_final : float  Max equiv. [G]0/[H]0
        dh         : float  Free host NMR resonance
        dhg        : float  Host-Guest complex NMR resonance
        dh2g       : float  Host-Guest2 complex NMR resonance
        num        : int    Number of steps to evaluate at

    Returns:
        (g0h0,     : tuple, length 5 of arrays of length num
         dd,         Simulated isotherm and molefraction curves
         mf_h,
         mf_hg,
         mf_h2g)

    Raises:
        [none]
    """

    # Convert all input to float64s
    k1         = np.float64(k1)
    k2         = np.float64(k2)
    h0_init    = np.float64(h0_init)
    g0h0_init  = np.float64(g0h0_init)
    g0h0_final = np.float64(g0h0_final)
    dh         = np.float64(dh)
    dhg        = np.float64(dhg)
    dh2g       = np.float64(dh2g)

    dtype = np.dtype('f8')

    # Initialise x array
    g0h0 = np.linspace(g0h0_init, g0h0_final, num, dtype=dtype)

    # Initialise y arrays
    shape = g0h0.shape
    dd = np.zeros(shape, dtype)       # Change in chemical shift delta (ppm or Hz)
    mf_h = np.zeros(shape, dtype)     # H free host molefraction
    mf_hg = np.zeros(shape, dtype)    # HG complex molefraction
    mf_h2g = np.zeros(shape, dtype)   # H2G complex molefraction

    for i, geq in enumerate(g0h0):
        # For convenience
        h0 = h0_init
        g0 = geq * h0

        # Calculate reduced cubic coefficients (for cubic in [H])
        a = 1
        b = 2*g0 - h0 + 1/k2;
        c = g0/k2 - h0/k2 + 1/(k1*k2);
        d = (-h0)/(k1*k2);

        p = np.array([a, b, c, d], dtype=dtype)

        # Find cubic roots (solve for [H])
        roots = np.roots(p).astype(dtype)

        # Find smallest real positive root:
        select = np.all([np.imag(roots) == 0, np.real(roots) >= 0], axis=0)
        h = roots[select].min()
        h = float(np.real(h))

        mf_hg[i]  = (g0*h*k1)/(h0*(1 + (k1*h) + (k2*k1*h*h)))
        mf_h2g[i] = (2*g0*k2*k1*h*h)/(h0*(1 + (k1*h) + (k2*k1*h*h)))
        mf_h[i]   = 1 - mf_hg[i] - mf_h2g[i]
        dd[i]     = dh*h0*mf_h[i] + dhg*h0*mf_hg[i] + dh2g*h0*mf_h2g[i]

    return g0h0, dd, mf_h, mf_hg, mf_h2g
