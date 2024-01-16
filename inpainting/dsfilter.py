# dsfilter.py

import taichi as ti
import inpainting

# Probably better to split into multiple kernels........
@ti.kernel
def step_DS_filter(
    u: ti.template(),
    dxy: ti.f32,
    u_DS: ti.template(),
    switch_DS: ti.template(),
    k_DS: ti.template(),
    radius_DS: ti.i32,
    λ: ti.f32,
    d_dx: ti.template(),
    d_dy: ti.template(),
    u_morph: ti.template(),
    switch_morph: ti.template(),
    k_morph: ti.template(),
    radius_morph: ti.i32,
    c: ti.template(),
    s: ti.template(),
    d_dxx: ti.template(),
    d_dxy: ti.template(),
    d_dyy: ti.template(),
    laplacian_u: ti.template(),
    dilation_u: ti.template(),
    dx_forward: ti.template(),
    dx_backward: ti.template(),
    dy_forward: ti.template(),
    dy_backward: ti.template(),
    dplus_forward: ti.template(),
    dplus_backward: ti.template(),
    dminus_forward: ti.template(),
    dminus_backward: ti.template(),
    abs_dx: ti.template(),
    abs_dy: ti.template(),
    abs_dplus: ti.template(),
    abs_dminus: ti.template()
):
    inpainting.switches.DS_switch(u_DS, k_DS, radius_DS, λ, d_dx, d_dy, switch_DS)
    inpainting.switches.morphological_switch(u_morph, k_morph, radius_morph, dxy, d_dx, d_dy, c, s, u, d_dxx, d_dxy, 
                                             d_dyy, switch_morph)
    inpainting.derivativesR2.laplacian(u, dxy, laplacian_u)
    inpainting.derivativesR2.dilation(u, dxy, dilation_u, dx_forward, dx_backward, dy_forward, dy_backward, 
                                      dplus_forward, dplus_backward, dminus_forward, dminus_backward, abs_dx, abs_dy, 
                                      abs_dplus, abs_dminus)
    
    for I in ti.grouped(u):
        u[I] = switch_DS[I] * laplacian_u[I] - (1 - switch_DS[I]) * switch_morph[I] * dilation_u[I]