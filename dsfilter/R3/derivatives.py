"""
    derivatives
    ===========

    Provides a variety of derivative operators on R^2, namely:
      1. `laplacian`: computes an approximation to the Laplacian with good
      rotation invariance, see Eq. (9) of [1] by K. Schaefer and J. Weickert.
      2. `morphological`: computes approximations to the dilation and erosion
      operators +/- ||grad u|| with good rotation invariance, see Eq. (12) of
      [1] by K. Schaefer and J. Weickert.

    References:
      [1]: K. Schaefer and J. Weickert.
      "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods in
      Computer Vision 14009 (2023), pp. 588--600.
      DOI:10.1137/15M1018460.
"""

import taichi as ti
from dsfilter.R3.utils import (
    sanitize_index,
    index_reflected
)
from dsfilter.utils import (
    select_upwind_derivative_dilation,
    select_upwind_derivative_erosion
)
# Actual Derivatives

@ti.kernel
def laplacian(
    u: ti.template(),
    dxy: ti.f32,
    laplacian_u: ti.template()
):
    """
    @taichi.kernel

    Compute an approximation of the Laplacian of `u` using axial and diagonal
    central differences, as described by by K. Schaefer and J. Weickert in
    Eq. (9) of [1].

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) which we want to
          differentiate.
        `dxy`: step size in x, y and z direction, taking values greater than 0.
        `dz`: step size z direction, taking values greater than 0.
      Mutated:
        `laplacian_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of laplacian of
          u, which is updated in place.

    References:
        [1]: K. Schaefer and J. Weickert.
          "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods
          in Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1137/15M1018460.
    """
    
    I_dx = ti.Vector([1, 0, 0], dt=ti.i32)
    I_dy = ti.Vector([0, 1, 0], dt=ti.i32)
    I_dz = ti.Vector([0, 0, 1], dt=ti.i32)
    for I in ti.grouped(laplacian_u):
        # We are on the grid, so interpolation is unnecessary, but I was lazy.

        # Axial Stencil
        # 0 |  1 | 0
        # 1 | -4 | 1
        # 0 |  1 | 0
        laplacian_u[I] = 1/ dxy**2 * (
            -4 * u[I] +
            index_reflected(u, I + I_dx) +
            index_reflected(u, I - I_dx) +
            index_reflected(u, I + I_dy) +
            index_reflected(u, I - I_dy) +
            index_reflected(u, I + I_dz) +
            index_reflected(u, I - I_dz)
        )

@ti.kernel
def morphological(
    u: ti.template(),
    dxy: ti.f32,
    dilation_u: ti.template(),
    erosion_u: ti.template()
):
    """
    @taichi.kernel

    Compute an approximation of the ||grad `u`|| using axial and diagonal upwind
    differences, as by K. Schaefer and J. Weickert in Eq. (12) of [1].

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `dilation_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of ||grad `u`||,
          which is updated in place.
        `erosion_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of -||grad `u`||,
          which is updated in place.
          
    References:
        [1]: K. Schaefer and J. Weickert.
          "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods
          in Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1137/15M1018460.
    """
    I_dx = ti.Vector([1, 0, 0], dt=ti.i32)
    I_dy = ti.Vector([0, 1, 0], dt=ti.i32)
    I_dz = ti.Vector([0, 0, 1], dt=ti.i32)
    for I in ti.grouped(dilation_u):

        # We are on the grid, so interpolation is unnecessary, but I was lazy.
        d_dx_forward = index_reflected(u, I + I_dx) - u[I]
        d_dx_backward = u[I] - index_reflected(u, I - I_dx)
        d_dy_forward = index_reflected(u, I + I_dy) - u[I]
        d_dy_backward = u[I] - index_reflected(u, I - I_dy)
        d_dz_forward = index_reflected(u, I + I_dz) - u[I]
        d_dz_backward = u[I] - index_reflected(u, I - I_dz)
        # Dilation
        ## Axial
        dilation_u[I] = 1 / dxy * ti.math.sqrt(
            select_upwind_derivative_dilation(d_dx_forward, d_dx_backward)**2 +
            select_upwind_derivative_dilation(d_dy_forward, d_dy_backward)**2 +
            select_upwind_derivative_dilation(d_dz_forward, d_dz_backward)**2
        )

        # Erosion
        ## Axial
        erosion_u[I] = -1 / dxy * ti.math.sqrt(
            select_upwind_derivative_erosion(d_dx_forward, d_dx_backward)**2 +
            select_upwind_derivative_erosion(d_dy_forward, d_dy_backward)**2 +
            select_upwind_derivative_erosion(d_dz_forward, d_dz_backward)**2
        )


@ti.func
def central_derivatives_first_order(
    u: ti.template(),
    dxy: ti.f32,
    d_dx: ti.template(),
    d_dy: ti.template(),
    d_dz: ti.template(),
):
    """
    @taichi.func

    Compute the second order derivatives of `u` using central differences.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `d_d**`: ti.field(dtype=[float], shape=[Nx, Ny]) of d* d* `u`, which is
          updated in place.
    """
    I_dx = ti.Vector([1, 0, 0], dt=ti.i32)
    I_dy = ti.Vector([0, 1, 0], dt=ti.i32)
    I_dz = ti.Vector([0, 0, 1], dt=ti.i32)
    for I in ti.grouped(u):
        I_dx_forward = sanitize_index(I + I_dx, u)
        I_dx_backward = sanitize_index(I - I_dx, u)
        I_dy_forward = sanitize_index(I + I_dy, u)
        I_dy_backward = sanitize_index(I - I_dy, u)
        I_dz_forward = sanitize_index(I + I_dz, u)
        I_dz_backward = sanitize_index(I - I_dz, u)

        d_dx[I] = (
            u[I_dx_forward] -
            
            u[I_dx_backward]
        ) / (dxy*2)

        d_dy[I] = (
            u[I_dy_forward] -
            u[I_dy_backward]
        ) / (dxy*2)

        d_dz[I] = (
            u[I_dz_forward] -
            u[I_dz_backward]
        ) / (dxy*2)

@ti.func
def central_derivatives_second_order(
    u: ti.template(),
    dxy: ti.f32,
    d_dxx: ti.template(),
    d_dxy: ti.template(),
    d_dyy: ti.template(),
    d_dxz: ti.template(),
    d_dyz: ti.template(),
    d_dzz: ti.template(),
):
    """
    @taichi.func

    Compute the second order derivatives of `u` using central differences.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `d_d**`: ti.field(dtype=[float], shape=[Nx, Ny]) of d* d* `u`, which is
          updated in place.
    """
    I_dx = ti.Vector([1, 0, 0], dt=ti.i32)
    I_dy = ti.Vector([0, 1, 0], dt=ti.i32)
    I_dz = ti.Vector([0, 0, 1], dt=ti.i32)
    I_dplus = I_dx + I_dy  # Positive diagonal
    I_dminus = I_dx - I_dy # Negative diagonal
    I_dplusxz = I_dx + I_dz  # Positive diagonal
    I_dminusxz = I_dx - I_dz # Negative diagonal
    I_dplusyz = I_dy + I_dz  # Positive diagonal
    I_dminusyz = I_dy - I_dz # Negative diagonal
    for I in ti.grouped(u):
        I_dx_forward = sanitize_index(I + I_dx, u)
        I_dx_backward = sanitize_index(I - I_dx, u)
        I_dy_forward = sanitize_index(I + I_dy, u)
        I_dy_backward = sanitize_index(I - I_dy, u)
        I_dz_forward = sanitize_index(I + I_dz, u)
        I_dz_backward = sanitize_index(I - I_dz, u)

        I_dplus_forward = sanitize_index(I + I_dplus, u)
        I_dplus_backward = sanitize_index(I - I_dplus, u)
        I_dminus_forward = sanitize_index(I + I_dminus, u)
        I_dminus_backward = sanitize_index(I - I_dminus, u)

        I_dplus_forwardxz = sanitize_index(I + I_dplusxz, u)
        I_dplus_backwardxz = sanitize_index(I - I_dplusxz, u)
        I_dminus_forwardxz = sanitize_index(I + I_dminusxz, u)
        I_dminus_backwardxz = sanitize_index(I - I_dminusxz, u)

        I_dplus_forwardyz = sanitize_index(I + I_dplusyz, u)
        I_dplus_backwardyz = sanitize_index(I - I_dplusyz, u)
        I_dminus_forwardyz = sanitize_index(I + I_dminusyz, u)
        I_dminus_backwardyz = sanitize_index(I - I_dminusyz, u)

        d_dxx[I] = (
            u[I_dx_forward] -
            u[I] * 2 +
            u[I_dx_backward]
        ) / dxy**2

        d_dxy[I] = (
            u[I_dplus_forward] -
            u[I_dminus_forward] -
            u[I_dminus_backward] +
            u[I_dplus_backward]
        ) / (4* dxy**2)

        d_dxz[I] = (
            u[I_dplus_forwardxz] -
            u[I_dminus_forwardxz] -
            u[I_dminus_backwardxz] +
            u[I_dplus_backwardxz]
        ) / (4* dxy**2)

        d_dyz[I] = (
            u[I_dplus_forwardyz] -
            u[I_dminus_forwardyz] -
            u[I_dminus_backwardyz] +
            u[I_dplus_backwardyz]
        ) / (4* dxy**2)

        d_dyy[I] = (
            u[I_dy_forward] -
            u[I] * 2 +
            u[I_dy_backward]
        ) / dxy**2

        d_dzz[I] = (
            u[I_dz_forward] -
            u[I] * 2 +
            u[I_dz_backward]
        ) / dxy**2