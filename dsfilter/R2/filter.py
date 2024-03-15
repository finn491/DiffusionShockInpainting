# dsfilter.py

import taichi as ti
import numpy as np
from tqdm import tqdm
from dsfilter.R2.switches import (
    DS_switch,
    morphological_switch
)
from dsfilter.R2.derivatives import (
    laplacian,
    morphological,
    gaussian_derivative_kernel
)
from dsfilter.utils import unpad_array

def DS_filter_R2(u0_np, mask_np, ν, λ, σ, ρ, dxy, T):
    """
    Apply Diffusion-Shock filtering in R^2.
    """
    # Set hyperparameters
    dt = compute_timestep(dxy)
    n = int(T / dt)
    k_DS, radius_DS = gaussian_derivative_kernel(ν, 0)
    k_morph_int, radius_morph_int = gaussian_derivative_kernel(σ, 0)
    k_morph_ext, radius_morph_ext = gaussian_derivative_kernel(ρ, 0)

    # Initialise TaiChi objects
    ## Padded versions of u to be able to do Gaussian derivative
    u0_np = np.pad(u0_np, pad_width=1, mode="reflect")
    shape = u0_np.shape
    u0 = ti.field(dtype=ti.f32, shape=shape)
    u0.from_numpy(u0_np)

    mask_np = np.pad(mask_np, pad_width=1, constant_values=1.)
    mask = ti.field(dtype=ti.f32, shape=shape)
    mask.from_numpy(mask_np)

    shape_padded = tuple(s + 2 for s in shape)
    u = ti.field(dtype=ti.f32, shape=shape_padded)
    fill_padded_u(u0, 2, u)
    du_dt = ti.field(dtype=ti.f32, shape=shape)

    shape_DS = tuple(s + 2 * radius_DS for s in shape)
    u_DS = ti.field(dtype=ti.f32, shape=shape_DS)
    fill_padded_u(u, radius_DS, u_DS)

    shape_morph_ext = tuple(s + 2 * (radius_morph_ext + radius_morph_int) for s in shape)
    u_morph_ext = ti.field(dtype=ti.f32, shape=shape_morph_ext)
    fill_padded_u(u, radius_morph_ext + radius_morph_int, u_morph_ext)
    shape_morph_σ_ext = tuple(s + 2 * radius_morph_ext for s in shape)
    u_morph_σ_ext = ti.field(dtype=ti.f32, shape=shape_morph_σ_ext)
    fill_padded_u(u, radius_morph_ext, u_morph_σ_ext)
    shape_morph_int = tuple(s + 2 * radius_morph_int for s in shape)
    u_morph_int = ti.field(dtype=ti.f32, shape=shape_morph_int)
    fill_padded_u(u, radius_morph_int, u_morph_int)

    ## Gaussian derivatives
    # shape_ext = (shape[0] + 2 * radius_morph_ext, shape[1] + 2 * radius_morph_ext)
    # d_dx = ti.field(dtype=ti.f32, shape=shape_ext)
    # d_dy = ti.field(dtype=ti.f32, shape=shape_ext)
    d_dx = ti.field(dtype=ti.f32, shape=shape)
    d_dy = ti.field(dtype=ti.f32, shape=shape)
    d_dxx = ti.field(dtype=ti.f32, shape=shape)
    d_dxy = ti.field(dtype=ti.f32, shape=shape)
    d_dyy = ti.field(dtype=ti.f32, shape=shape)

    ## Dominant eigenvector
    Jρ_padded = ti.field(dtype=ti.f32, shape=shape_morph_σ_ext)
    Jρ11 = ti.field(dtype=ti.f32, shape=shape)
    Jρ12 = ti.field(dtype=ti.f32, shape=shape)
    Jρ22 = ti.field(dtype=ti.f32, shape=shape)
    c = ti.field(dtype=ti.f32, shape=shape)
    s = ti.field(dtype=ti.f32, shape=shape)

    ## Switch between diffusion and shock and between dilation and erosion
    switch_DS = ti.field(dtype=ti.f32, shape=shape)
    switch_morph = ti.field(dtype=ti.f32, shape=shape)

    ## Sobel derivatives
    laplacian_u = ti.field(dtype=ti.f32, shape=shape)
    dilation_u = ti.field(dtype=ti.f32, shape=shape)
    erosion_u = ti.field(dtype=ti.f32, shape=shape)

    for _ in tqdm(range(n)):
        # Compute switches
        DS_switch(u_DS, dxy, k_DS, radius_DS, λ, d_dx, d_dy, switch_DS)
        morphological_switch(u_morph_ext, u_morph_σ_ext, u_morph_int, dxy, k_morph_int, radius_morph_int, d_dx, d_dy,
                             k_morph_ext, radius_morph_ext, Jρ_padded, Jρ11, Jρ12, Jρ22, c, s, d_dxx, d_dxy, d_dyy,
                             switch_morph)
        # Compute derivatives
        laplacian(u, dxy, laplacian_u)
        morphological(u, dxy, dilation_u, erosion_u)
        # Step
        step_DS_filter(u, mask, dt, switch_DS, switch_morph, laplacian_u, dilation_u, erosion_u, du_dt)
        # Correct boundary conditions
        fix_reflected_padding(u, 1)
        fill_padded_u(u, radius_morph_ext + radius_morph_int, u_morph_ext)
        fix_reflected_padding(u_morph_ext, radius_morph_ext + radius_morph_int)
        fill_padded_u(u, radius_morph_ext, u_morph_σ_ext)
        fix_reflected_padding(u_morph_σ_ext, radius_morph_ext)
        fill_padded_u(u, radius_morph_int, u_morph_int)
        fill_padded_u(u, radius_morph_int, u_morph_int)
    # Cleanup   
    u_np = u.to_numpy()
    return unpad_array(u_np, pad_shape=1)

def compute_timestep(dxy, δ=np.sqrt(2)-1):
    """
    Compute timestep to solve Diffusion-Shock PDE, such that the scheme retains
    the maximum-minimum principle of the continuous PDE.
    
    Args:
        `dxy`: step size in x and y direction, taking values greater than 0.
      Optional:
        `δ`: weight parameter to balance axial vs diagonal finite difference
          schemes, taking values between 0 and 1. Defaults to `np.sqrt(2)-1`,
          which leads to good rotation invariance according to
          "PDE Evolutions for M-Smoothers in One, Two, and Three Dimensions"
          (2020) by M. Welk and J. Weickert.
    
    Returns:
        timestep, taking values greater than 0.
    """
    τ_D = dxy**2 / (4 - 2 * δ)
    τ_M = dxy / (np.sqrt(2) * (1 - δ) + δ)
    return min(τ_D, τ_M) / 2


@ti.kernel
def step_DS_filter(
    u: ti.template(),
    mask: ti.template(),
    dt: ti.f32,
    switch_DS: ti.template(),
    switch_morph: ti.template(),
    laplacian_u: ti.template(),
    dilation_u: ti.template(),
    erosion_u: ti.template(),
    du_dt: ti.template()
):
    """
    @taichi.kernel

    Perform a single timestep "Diffusion-Shock Inpainting" (2023) by K.
    Schaefer and J. Weickert, Eq. (12).

    Args:
      Static:
        `mask`: ti.field(dtype=[float], shape=[Nx, Ny]) inpainting mask.
        `dt`: step size, taking values greater than 0.
        `switch_DS`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of values that
          determine the degree of diffusion or shock, taking values between 0
          and 1.
        `switch_morph`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of values that
          determine the degree of dilation or erosion, taking values between -1
          and 1.
        `laplacian_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of laplacian of
          `u`, which is updated in place.
        `dilation_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of ||grad `u`||,
          which is updated in place.
        `erosion_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of -||grad `u`||,
          which is updated in place.
      Mutated:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) which we want to evolve
          with the DS PDE.
        `du_dt`: ti.field(dtype=[float], shape=[Nx, Ny]) change in `u` in a
          single time step, not taking into account the mask.
    """
    for I in ti.grouped(u):
        du_dt[I] = (
            laplacian_u[I] * switch_DS[I] - 
            (1 - switch_DS[I]) * (
                # Do erosion when switch_morph = 1.
                erosion_u[I] * (1 + switch_morph[I]) / 2  +
                # Do dilation when switch_morph = -1.
                dilation_u[I] * (1 - switch_morph[I]) / 2
            )
        )
        u[I] += dt * du_dt[I] * (1 - mask[I]) # Only change values in the mask.
        

# Fix padding function
        
@ti.kernel
def fix_reflected_padding(
    u: ti.template(),
    radius: ti.i32
):
    """
    @taichi.kernel

    Repad so that to satisfy reflected boundary conditions.
    
    Args:
      Mutated:
        `u`: ti.field(dtype=[float], shape=[Nx+2*`radius`, Ny+2*`radius`]) to be
          repadded, updated in place.
    """
    I, J = u.shape
    for i in range(I):
        for k in range(radius):
            u[i, k] = u[i, 2 * radius - k]
            u[i, J-1 - k] = u[i, J-1 - 2 * radius + k]
    for j in range(J):
        for k in range(radius):
            u[k, j] = u[2 * radius - k, j]
            u[I-1 - k, j] = u[I-1 - 2 * radius + k, j]

@ti.kernel
def fill_padded_u(
    u: ti.template(),
    radius: ti.i32,
    padded_u: ti.template()
):
    """
    @taichi.kernel

    Update the content of the field used to determine the switch.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx+2, Ny+2]) content to fill
          `padded_u`.
        `radius`: radius of the Gaussian filter used in the switch, taking
          integer values greater than 0.
      Mutated:
        `padded_u`: ti.field(dtype=[float], shape=[Nx+2*`radius`, Ny+2*`radius`])
          switch, updated in place.
    """
    I_shift = ti.Vector([radius - 1, radius - 1], ti.i32)
    for I in ti.grouped(u):
        padded_u[I + I_shift] = u[I]