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
    dilation,
    gaussian_derivative_kernel
)
from dsfilter.R2.metric import (
    align_to_real_axis_scalar_field,
    align_to_standard_array_axis_scalar_field
)
from dsfilter.utils import unpad_array

def DS_filter_R2(u0_np, mask_np, ν, λ, σ, dxy, T):
    # Align with (x, y)-frame
    u0_np = align_to_real_axis_scalar_field(u0_np)
    mask_np = align_to_real_axis_scalar_field(mask_np)
    # shape = u0.shape

    # Set hyperparameters
    dt = compute_timestep(dxy)
    n = int(T / dt)
    k_DS, radius_DS = gaussian_derivative_kernel(ν, 1)
    k_morph, radius_morph = gaussian_derivative_kernel(σ, 1)

    # Initialise TaiChi objects
    ## Padded versions of u to be able to do Gaussian derivative
    u0_np = np.pad(u0_np, pad_width=1, mode="reflect")
    shape = u0_np.shape
    u0 = ti.field(dtype=ti.f32, shape=shape)
    u0.from_numpy(u0_np)

    mask_np = np.pad(mask_np, pad_width=1, constant_values=1.)
    mask = ti.field(dtype=ti.f32, shape=shape)
    mask.from_numpy(mask_np)

    u = ti.field(dtype=ti.f32, shape=shape)
    u.from_numpy(u0_np)

    u_DS = ti.field(dtype=ti.f32, shape=shape)
    u_DS.from_numpy(u0_np)

    u_morph = ti.field(dtype=ti.f32, shape=shape)
    u_morph.from_numpy(u0_np)

    ## Gaussian derivatives
    d_dx = ti.field(dtype=ti.f32, shape=shape)
    d_dy = ti.field(dtype=ti.f32, shape=shape)
    d_dxx = ti.field(dtype=ti.f32, shape=shape)
    d_dxy = ti.field(dtype=ti.f32, shape=shape)
    d_dyy = ti.field(dtype=ti.f32, shape=shape)

    ## Dominant eigenvector
    c = ti.field(dtype=ti.f32, shape=shape)
    s = ti.field(dtype=ti.f32, shape=shape)

    ## Switch between diffusion and shock and between dilation and erosion
    switch_DS = ti.field(dtype=ti.f32, shape=shape)
    switch_morph = ti.field(dtype=ti.f32, shape=shape)

    ## Sobel derivatives
    laplacian_u = ti.field(dtype=ti.f32, shape=shape)
    dilation_u = ti.field(dtype=ti.f32, shape=shape)
    dx_forward = ti.field(dtype=ti.f32, shape=shape)
    dx_backward = ti.field(dtype=ti.f32, shape=shape)
    dy_forward = ti.field(dtype=ti.f32, shape=shape)
    dy_backward = ti.field(dtype=ti.f32, shape=shape)
    dplus_forward = ti.field(dtype=ti.f32, shape=shape)
    dplus_backward = ti.field(dtype=ti.f32, shape=shape)
    dminus_forward = ti.field(dtype=ti.f32, shape=shape)
    dminus_backward = ti.field(dtype=ti.f32, shape=shape)
    abs_dx = ti.field(dtype=ti.f32, shape=shape)
    abs_dy = ti.field(dtype=ti.f32, shape=shape)
    abs_dplus = ti.field(dtype=ti.f32, shape=shape)
    abs_dminus = ti.field(dtype=ti.f32, shape=shape)

    for _ in tqdm(range(n)):
        DS_switch(u_DS, k_DS, radius_DS, λ, d_dx, d_dy, switch_DS)
        morphological_switch(u_morph, k_morph, radius_morph, dxy, d_dx, d_dy, c, s, u, d_dxx, d_dxy, d_dyy,
                             switch_morph)
        laplacian(u, dxy, laplacian_u)
        dilation(u, dxy, dx_forward, dx_backward, dy_forward, dy_backward, dplus_forward, dplus_backward,
                 dminus_forward, dminus_backward, abs_dx, abs_dy, abs_dplus, abs_dminus, dilation_u) 
        step_DS_filter(u, dt, switch_DS, switch_morph, laplacian_u, dilation_u)
        apply_mask(u, u0, mask)
        fix_reflected_padding(u)
        fix_switch_padding(u, radius_DS, u_DS)
        fix_switch_padding(u, radius_morph, u_morph)

    # Align with (I, J)-frame    
    u_np = u.to_numpy()
    u_np = unpad_array(u_np, pad_shape=1)
    u_np = align_to_standard_array_axis_scalar_field(u_np)
    return u_np

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
    dt: ti.f32,
    switch_DS: ti.template(),
    switch_morph: ti.template(),
    laplacian_u: ti.template(),
    dilation_u: ti.template()
):
    """
    @taichi.kernel

    Perform a single timestep "Diffusion-Shock Inpainting" (2023) by K.
    Schaefer and J. Weickert, Eq. (12).

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `dilation_u`: ti.field(dtype=[float], shape=shape) of ||grad `u`||,
          which is updated in place.
    """
    for I in ti.grouped(u):
        u[I] += dt * (switch_DS[I] * laplacian_u[I] - (1 - switch_DS[I]) * switch_morph[I] * dilation_u[I])

# Fix padding function
        
@ti.kernel
def fix_reflected_padding(
    u: ti.template()
):
    """fix the padding"""
    I, J = u.shape
    for i in range(I):
        u[i, 0] = u[i, 2]
        u[i, J-1] = u[i, J-3]
    for j in range(J):
        u[0, j] = u[2, j]
        u[I-1, j] = u[I-3, j]

@ti.kernel
def fix_switch_padding(
    u: ti.template(),
    radius: ti.i32,
    switch: ti.template()
):
    I_shift = ti.Vector([radius - 1, radius - 1], ti.i32)
    for I in ti.grouped(u):
        switch[I + I_shift] = u[I]

@ti.kernel
def apply_mask(
    u: ti.template(),
    u0: ti.template(),
    mask: ti.template()
):
    for I in ti.grouped(u):
        u[I] = (1 - mask[I]) * u[I] + mask[I] * u0[I]