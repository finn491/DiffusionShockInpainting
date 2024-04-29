"""
    orientationscore
    ================

    Provides methods to compute the orientation score of a 2D image. The primary
    methods are:
      1. `cakewavelet_stack`: compute the stack of cakewavelets, as described in
      Duits "Perceptual Organization in Image Analysis" (2005)
      (https://www.win.tue.nl/~rduits/THESISRDUITS.pdf).
      2. `wavelet_transform`: compute the real wavelet transform of a 2D image,
      with respect to some wavelet (which need not be a cakewavelet).
      3. `wavelet_transform_complex`: compute the complex wavelet transform of a
      2D image, with respect to some wavelet (which need not be a cakewavelet).
      TODO: figure out why `wavelet_transform_complex` does not work...
"""

import numpy as np
import scipy as sp

def mod_offset(x, period, offset):
    """Compute `x` modulo `period` with offset `offset`."""
    return x - (x - offset)//period * period

def rotate_left(array, k):
    """Rotate left the columns and rows of `array` by `k`."""
    return rotate_right(array, -k)

def rotate_right(array, k):
    """Rotate right the columns and rows of `array` by `k`."""
    if type(k) == int or type(k) == float:
        arr1 = array[:-k]
        arr2 = array[-k:]
        rotated_array = np.concatenate((arr2, arr1), axis=0)
    elif len(k) == 2 and len(array.shape) == 2:
        k_0, k_1 = k
        arr1 = array[:, :-k_1]
        arr2 = array[:, -k_1:]
        array = np.concatenate((arr2, arr1), axis=1)
        arr1 = array[:-k_0, :]
        arr2 = array[-k_0:, :]
        rotated_array = np.concatenate((arr2, arr1), axis=0)
    else:
        raise ValueError(f"k = {k} is not a supported point for rotating. k should be a number or a pair of numbers.")
    return rotated_array

def Gauss_window(N_spatial, σ_s):
    """
    Compute a Gaussian envelope, which can be used as a low pass filter by 
    multiplying pointwise in the Fourier domain.
    """
    xs, ys = np.meshgrid(np.arange(-np.floor(N_spatial / 2), np.ceil(N_spatial / 2)),
                         np.arange(-np.floor(N_spatial / 2), np.ceil(N_spatial / 2)),
                         indexing="ij")
    out = np.exp(-(xs**2 + ys**2) / (2 * σ_s**2))
    return out

def angular_grid(N_spatial):
    """Compute a grid of angle coordinates."""
    centerx = np.ceil((N_spatial - 1) / 2)
    centery = centerx
    xs, ys = np.meshgrid(np.arange(N_spatial), np.arange(N_spatial), indexing="ij")
    dxs = xs - centerx
    dys = ys - centery
    θs = np.arctan2(dys, dxs)
    return θs

def radial_grid(N_spatial):
    """Compute a grid of radial coordinates."""
    centerx = N_spatial // 2
    centery = centerx
    xs, ys = np.meshgrid(np.arange(N_spatial), np.arange(N_spatial), indexing="ij")
    dxs = xs - centerx
    dys = ys - centery
    rs = 2 * np.sqrt(dxs**2 + dys**2) / N_spatial #  + np.finfo(np.float64).eps)
    return rs

def radial_window(N_spatial, n, inflection_point):
    """
    Compute a smooth radial window in the Fourier domain for limiting the
    bandwidth of the cakewavelets.

    Corresponds to M_N, given by Eq. (4.41) in Duits
    "Perceptual Organization in Image Analysis" (2005).
    """
    ε = np.finfo(np.float64).eps
    ρ_matrix = ε + radial_grid(N_spatial) / np.sqrt(2 * inflection_point**2 / (1 + 2 * n))
    s = np.zeros_like(ρ_matrix)
    exp_ρ_squared = np.exp(-ρ_matrix**2)
    for k in range(n + 1):
        s = s + exp_ρ_squared * ρ_matrix**(2*k) / sp.special.factorial(k)
    return s

def B_spline_matrix(n, x):
    """
    Compute degree `n` B-splines.

    In this way, the sum of all cakewavelets in the Fourier domain is
    identically equal to 1 (within the disk M), while each cakewavelet varies
    smoothly in the angular direction in the Fourier domain. See Section 4.6
    in Duits "Perceptual Organization in Image Analysis" (2005).
    """
    # This is the bottleneck of computing the cakewavelet stack.
    ε = np.finfo(np.float64).eps
    # Only need to compute these coefficients once.
    coeffs = []
    for k in range(n + 2):
        binom_cof = sp.special.binom(n + 1, k)
        coeffs.append(binom_cof * (x + (n + 1) / 2 - k) ** n * (-1)**k)

    r = 0
    for i in np.arange(-n/2, n/2 + 1):
        s = 0
        # There seems to be no way to do this without a loop that does not break
        # broadcasting, except for allocating meshgrid arrays, which is slower.
        for k in range(n + 2):
            sign = np.sign(i + (n + 1) / 2 - k)
            s += coeffs[k] * sign

        f = s / (2 * sp.special.factorial(n))
        interval_check = (x >= (i - 1/2 + ε)) * (x <= (i + 1/2 - ε * (i >= n / 2)))                     
        r += f * np.round(interval_check)
    return r

def cakewavelet_stack_fourier(N_spatial, dθ, spline_order, overlap_factor, inflection_point, mn_order):
    """
    Compute the cakewavelets in the Fourier domain.

    Args:
        `N_spatial`: number of pixels in each spatial direction. This notably
          means that the support of the wavelets will be a square.
        `dθ`: angular resolution in radians.
        `spline_order`: degree of the B-splines.
        `overlap_factor`: degree to which adjacent slices overlap in the
          angular direction. When `overlap_factor` is larger than 1, then
          multiple wavelets will cover the same angles.
        `inflection_point`: point at which the radial window M_N starts to
          decrease, taking values at most 1. By increasing this will improve the
          stability of the reconstruction, but the L^1 norm of the cakewavelets
          will also increase.
        `mn_order`: order at which the geometric sum in the radial window is
          truncated.
        `DC_σ`: standard deviation of the high pass filter used to remove the
          DC component, such that the cakewavelets can be constructed around
          the origin in the Fourier domain.
    """
    mn_window = radial_window(N_spatial, mn_order, inflection_point)
    window =  mn_window
    angle_grid = angular_grid(N_spatial)
    dθ_overlapped = dθ / overlap_factor
    s = 2 * np.pi
    θs = np.arange(0, s, dθ_overlapped)
    filters = np.zeros((θs.shape[0], N_spatial, N_spatial))
    for i, θ in enumerate(θs):
        x = mod_offset(angle_grid - θ - np.pi / 2, 2 * np.pi, -np.pi) / dθ
        filters[i] = window * B_spline_matrix(spline_order, x)
    return filters

def cakewavelet_stack(N_spatial, Nθ, inflection_point=0.8, mn_order=8, spline_order=3, overlap_factor=1,
                      Gaussian_σ=None):
    """
    Compute the cakewavelets in the Fourier domain.

    Args:
        `N_spatial`: number of pixels in each spatial direction. This notably
          means that the support of the wavelets will be a square.
        `Nθ`: number of orientations.
      Optional:
        `inflection_point`: point at which the radial window M_N starts to
          decrease, taking values at most 1. By increasing this will improve the
          stability of the reconstruction, but the L^1 norm of the cakewavelets
          will also increase. Defaults to 0.8
        `mn_order`: order at which the geometric sum in the radial window is
          truncated. Defaults to 10.
        `spline_order`: degree of the B-splines. Defaults to 3.
        `overlap_factor`: degree to which adjacent slices overlap in the
          angular direction. When `overlap_factor` is larger than 1, then
          multiple wavelets will cover the same angles. Defaults to 1.
        `Gaussian_σ`: standard deviation of the Gaussian window that is applied
          to remove the long tails of the cakewavelets. Defaults to
          (`N_spatial` - 1) / 4.
    """
    # Set default values
    if Gaussian_σ is None:
        Gaussian_σ = (N_spatial - 1) / 4
    dθ = 2 * np.pi / Nθ
    cake_fourier = cakewavelet_stack_fourier(N_spatial, dθ, spline_order, overlap_factor, inflection_point, mn_order)

    cake_fourier[:, (N_spatial//2 - 2):(N_spatial//2 + 3), (N_spatial//2 - 2):(N_spatial//2 + 3)] = dθ / (2 * np.pi)

    cake = np.zeros_like(cake_fourier, dtype=np.complex_)
    rotation_amounts = np.array((N_spatial // 2, N_spatial // 2))
    window = Gauss_window(N_spatial, Gaussian_σ)
    for i, slice_fourier in enumerate(cake_fourier):
        slice_fourier = rotate_left(slice_fourier, rotation_amounts)
        # Mathematica uses Fourier parameters {a, b} = {0, 1} by default, while
        # NumPy uses {a, b} = {1, -1}. The inverse Fourier transform is then
        # given by (http://reference.wolfram.com/language/ref/InverseFourier.html)
        #   sum_s^n ν_s exp(-2πi b (r - 1) (s - 1) / n) / n^((1 + a)/2).
        # Hence, Mathematica computes
        #   sum_s^n ν_s exp(-2πi (r - 1) (s - 1) / n) / n^(1/2),
        # while NumPy computes
        #   sum_s^n ν_s exp(2πi (r - 1) (s - 1) / n) / n.
        # Therefore, we can get Mathematica's result by conjugating and
        # multiplying by n^(1/2).
        # However, if we use NumPy's convention, we can forget about n when
        # performing convolutions, so we don't multiply by n^(1/2).
        slice = np.conj(np.fft.ifftn(slice_fourier))
        slice = rotate_right(slice, rotation_amounts)
        cake[i] = slice * window

    return cake
    
def wavelet_transform(f, kernels):
    """
    Return the real part of the wavelet transform of image `f` under the
    `kernels`.
    """
    N_spatial = f.shape[0]
    ost = np.zeros((kernels.shape[0], N_spatial, N_spatial))
    f_hat = np.fft.fftn(f)
    rotation_amount = np.floor(0.1 + np.array(f.shape) / 2).astype(int) # Why?
    for i, ψ_θ in enumerate(kernels):
        ψ_θ_hat = np.fft.fftn(ψ_θ)
        U_θ_hat = np.fft.ifftn(ψ_θ_hat * f_hat).real
        U_θ_hat = rotate_right(U_θ_hat, rotation_amount)
        ost[i] = U_θ_hat
    return ost

def wavelet_transform_complex(f, kernels):
    """
    Return the wavelet transform of image `f` under the `kernels`.
    """
    N_spatial = f.shape[0]
    ost = np.zeros((kernels.shape[0], N_spatial, N_spatial), dtype=np.complex_)
    f_hat = np.fft.fftn(f)
    rotation_amount = np.floor(0.1 + np.array(f.shape) / 2).astype(int) # Why?
    for i, ψ_θ in enumerate(kernels):
        ψ_θ_hat = np.fft.fftn(ψ_θ)
        U_θ_hat = np.fft.ifftn(ψ_θ_hat * f_hat)
        U_θ_hat = rotate_right(U_θ_hat, rotation_amount)
        ost[i] = U_θ_hat
    return ost