"""
    utils
    =====

    Provides miscellaneous computational utilities that can be used both on R^2
    and SE(2).
"""

import numpy as np
import taichi as ti


# Interpolation

@ti.func
def linear_interpolate(
    v0: ti.f32,
    v1: ti.f32,
    r: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Interpolate value of the points `v*` depending on the distance `r`, via 
    linear interpolation. Adapted from Gijs.

    Args:
        `v*`: values at points between which we want to interpolate, taking real 
          values.
        `r`: distance to the points between which we to interpolate, taking real
          values.

    Returns:
        Interpolated value.
    """
    return v0 * (1.0 - r) + v1 * r

# Derivatives

@ti.func
def select_upwind_derivative_dilation(
    d_forward: ti.f32,
    d_backward: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Select the correct derivative for the upwind derivative of dilation.

    Args:
        `d_forward`: derivative in the forward direction.
        `d_backward`: derivative in the backward direction.
          
    Returns:
        derivative in the correct direction.
    """
    return ti.math.max(d_forward, -d_backward, 0) * (-1.)**(d_forward <= -d_backward)

@ti.func
def select_upwind_derivative_erosion(
    d_forward: ti.f32,
    d_backward: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Select the correct derivative for the upwind derivative of erosion.

    Args:
        `d_forward`: derivative in the forward direction.
        `d_backward`: derivative in the backward direction.
          
    Returns:
        derivative in the correct direction.
    """
    return ti.math.max(-d_forward, d_backward, 0) * (-1.)**(-d_forward >= d_backward)

# Switches

@ti.func
def S_ε(
    x: ti.f32,
    ε: ti.f32
) -> ti.f32:
    """
    @taichi.func
    
    Compute Sε, the regularised signum as given by K. Schaefer and J. Weickert
    in Eq (7) in [1].

    Args:
        `x`: scalar to pass through regularised signum, taking values greater 
          than 0.
        `ε`: regularisation parameter, taking values greater than 0.

    Returns:
        ti.f32 of S`ε`(`x`).

    References:
        [1]: K. Schaefer and J. Weickert.
          "Regularised Diffusion-Shock Inpainting". In: Journal of Mathematical
          Imaging and Vision (2024).
          DOI:10.1007/s10851-024-01175-0.
    """
    return (2 / ti.math.pi) * ti.math.atan2(x, ε)

@ti.func
def g_scalar(
    s_squared: ti.f32, 
    λ: ti.f32
) -> ti.f32:
    """
    @taichi.func
    
    Compute g, the function that switches between diffusion and shock, given
    by Eq. (5) in [1] by K. Schaefer and J. Weickert.

    Args:
        `s_squared`: square of some scalar, taking values greater than 0.
        `λ`: contrast parameter, taking values greater than 0.

    Returns:
        ti.f32 of g(`s_squared`).

    References:
        [1]: K. Schaefer and J. Weickert.
          "Regularised Diffusion-Shock Inpainting". In: Journal of Mathematical
          Imaging and Vision (2024).
          DOI:10.1007/s10851-024-01175-0.
    """
    return 1 / ti.math.sqrt(1 + s_squared / λ**2)

# Padding

def pad_array(u, pad_value=0., pad_shape=1):
    """
    Pad a numpy array `u` in each direction with `pad_shape` entries of values 
    `pad_value`.

    Args:
        `u`: np.ndarray that is to be padded.
        `pad_value`: Value with which `u` is padded.
        `pad_shape`: Number of cells that is padded in each direction, taking
          positive integral values.

    Returns:
        np.ndarray of the padded array.
    """
    return np.pad(u, pad_width=pad_shape, constant_values=pad_value)


def extract_centre_slice(u, pad_shape=1):
    """
    Select the centre part of `u`, removing padding of size `pad_shape` in each 
    direction.

    Args:
        `u`: np.ndarray from which the centre is to be extracted.
        `pad_shape`: Number of cells that is padded in each direction.

    Returns:
        np.ndarray of the unpadded array.
    """
    if type(pad_shape) == int:
        centre_slice = tuple(slice(pad_shape, dim_len - pad_shape, 1) for dim_len in u.shape)
    else: 
        centre_slice = tuple(slice(pad_shape[i], dim_len - pad_shape[i], 1) for i, dim_len in enumerate(u.shape))
    return centre_slice


def unpad_array(u, pad_shape=1):
    """
    Remove the outer `pad_shape` entries numpy array `u`.

    Args:
        `u`: np.ndarray from which the centre is to be extracted.
        `pad_shape`: Number of cells that is padded in each direction, taking
          positive integral values.

    Returns:
        np.ndarray of the unpadded array. Note that 
          `unpad_array(*, pad_shape=pad_shape)` is the left-inverse of
          `pad_array(*, pad_value=pad_value, pad_shape=pad_shape)` for all
          `pad_value`, `pad_shape`.
    """
    centre_slice = extract_centre_slice(u, pad_shape=pad_shape)
    return u[centre_slice]

# Initialisation

def get_padded_cost(cost_unpadded, pad_shape=1, pad_value=1.):
    """Pad the cost function `cost_unpadded` and convert to TaiChi object."""
    cost_np = pad_array(cost_unpadded, pad_value=pad_value, pad_shape=pad_shape)
    cost = ti.field(dtype=ti.f32, shape=cost_np.shape)
    cost.from_numpy(cost_np)
    return cost

def get_initial_W(shape, initial_condition=100., pad_shape=1):
    """Initialise the (approximate) distance map as TaiChi object."""
    W_unpadded = np.full(shape=shape, fill_value=initial_condition)
    W_np = pad_array(W_unpadded, pad_value=initial_condition, pad_shape=pad_shape)
    W = ti.field(dtype=ti.f32, shape=W_np.shape)
    W.from_numpy(W_np)
    return W

@ti.kernel
def apply_boundary_conditions(
    u: ti.template(), 
    boundarypoints: ti.template(), 
    boundaryvalues: ti.template()
):
    """
    @taichi.kernel

    Apply `boundaryvalues` at `boundarypoints` to `u`.

    Args:
      Static:
        `boundarypoints`: ti.Vector.field(n=dim, dtype=[int], shape=N_points),
          where `N_points` is the number of boundary points and `dim` is the 
          dimension of `u`.
        `boundaryvalues`: ti.Vector.field(n=dim, dtype=[float], shape=N_points),
          where `N_points` is the number of boundary points and `dim` is the 
          dimension of `u`.
      Mutated:
        `u`: ti.field(dtype=[float]) to which the boundary conditions should be 
          applied.
    """
    for I in ti.grouped(boundarypoints):
        u[boundarypoints[I]] = boundaryvalues[I]

# Image Preprocessing
        
def image_rescale(image_array, new_max=1.):
    """
    Affinely rescale values in numpy array `image_array` to be between 0. and
    `new_max`.
    """
    image_max = image_array.max()
    image_min = image_array.min()
    return new_max * (image_array - image_min) / (image_max - image_min)

# Quality Measures

@ti.kernel
def PSNR(
    u: ti.template(),
    ground_truth: ti.template(),
    MAX: ti.f32
) -> ti.f32:
    """
    @taichi.kernel

    Compute the Peak Signal-to-Noise Ratio (PSNR) of `u` with respect to
    `ground_truth`.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) reconstruction of image.
        `ground_truth`: ti.field(dtype=[float], shape=[Nx, Ny]), image that is
          being reconstructed.
        `MAX`: maximum possible value a pixel can take. Typically 1 or 255.

    Returns:
        PSNR in dB of `u` with respect to `ground_truth`.
    """
    # Total squared error/L2 error
    SE = 0
    for I in ti.grouped(u):
        SE += (u[I] - ground_truth[I])**2
    # Mean squared error
    shape = u.shape
    N = shape[0] * shape[1]
    MSE = SE / N
    # PSNR is in dB, so we log_10; log_10(x) = ln(x) / ln(10).
    ln_10 = ti.static(ti.math.log(10))
    return 10 * ti.math.log((MAX**2) / MSE) / ln_10
