# dsfilter.py

import taichi as ti
import numpy as np
import inpainting


def DS_filter_R2(u0, ν, λ, σ, dxy, T):
    dt = compute_timestep(dxy)
    n = int(T / dt)
    k_DS, radius_DS = inpainting.derivativesR2.gaussian_derivative_kernel(ν, 1)
    k_morph, radius_morph = inpainting.derivativesR2.gaussian_derivative_kernel(σ, 1)
    for _ in range(n):
        inpainting.switches.DS_switch(u_DS, k_DS, radius_DS, λ, d_dx, d_dy, switch_DS)
        inpainting.switches.morphological_switch(u_morph, k_morph, radius_morph, dxy, d_dx, d_dy, c, s, u, d_dxx, d_dxy, 
                                                 d_dyy, switch_morph)
        inpainting.derivativesR2.laplacian(u, dxy, laplacian_u)
        inpainting.derivativesR2.dilation(u, dxy, dx_forward, dx_backward, dy_forward, dy_backward, dplus_forward, 
                                          dplus_backward, dminus_forward, dminus_backward, abs_dx, abs_dy, abs_dplus, 
                                          abs_dminus, dilation_u) 
        step_DS_filter(u, switch_DS, switch_morph, laplacian_u, dilation_u)
        # Deal with BCs.

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
        `dilation_u`: ti.field(dtype=[float], shape=shape) of |grad `u`|,
          which is updated in place.
    """
    for I in ti.grouped(u):
        u[I] = dt * (switch_DS[I] * laplacian_u[I] - (1 - switch_DS[I]) * switch_morph[I] * dilation_u[I])