# dsfilter.py

import taichi as ti
import inpainting


def DS_filter_R2(u0, ν, λ, σ, T):
    
    k_DS, radius_DS = compute_gaussian_derivative_kernel(ν, 1)
    k_morph, radius_morph = compute_gaussian_derivative_kernel(σ, 1)
    for t in range(T):
        inpainting.switches.DS_switch(u_DS, k_DS, radius_DS, λ, d_dx, d_dy, switch_DS)
        inpainting.switches.morphological_switch(u_morph, k_morph, radius_morph, dxy, d_dx, d_dy, c, s, u, d_dxx, d_dxy, 
                                                 d_dyy, switch_morph)
        inpainting.derivativesR2.laplacian(u, dxy, laplacian_u)
        inpainting.derivativesR2.dilation(u, dxy, dx_forward, dx_backward, dy_forward, dy_backward, dplus_forward, dplus_backward, 
                           dminus_forward, dminus_backward, abs_dx, abs_dy, abs_dplus, abs_dminus, dilation_u) 
        step_DS_filter(u, switch_DS, switch_morph, laplacian_u, dilation_u)
        # Deal with BCs.

def compute_gaussian_derivative_kernel(σ, order, truncate=5., dxy=1.):
    radius = int(σ * truncate + 0.5)
    k = ti.field(dtype=ti.f32, shape=2*radius+1)
    inpainting.derivativesR2.gaussian_derivative_kernel(σ, order, radius, dxy, k)
    return k, radius


@ti.kernel
def step_DS_filter(
    u: ti.template(),
    switch_DS: ti.template(),
    switch_morph: ti.template(),
    laplacian_u: ti.template(),
    dilation_u: ti.template()
):
    for I in ti.grouped(u):
        u[I] = switch_DS[I] * laplacian_u[I] - (1 - switch_DS[I]) * switch_morph[I] * dilation_u[I]