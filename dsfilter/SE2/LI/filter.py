"""
    filter
    ======

    Provides methods to apply SE(2) Diffusion-Shock inpainting, inspired by the
    Diffusion-Shock inpainting on R^2 by K. Schaefer and J. Weickert.[1][2]
    The primary methods are:
      1. `DS_filter_lines`: apply SE(2) Diffusion-Shock inpainting to an array
      describing an image consisting of lines, given an inpainting mask and
      various PDE parameters.
      1. `DS_filter_planes`: apply SE(2) Diffusion-Shock inpainting to an array
      describing an image consisting of lines and flat areas, given an
      inpainting mask and various PDE parameters.

    References:
      [1]: K. Schaefer and J. Weickert.
      "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods in
      Computer Vision 14009 (2023), pp. 588--600.
      DOI:10.1137/15M1018460.
      [2]: K. Schaefer and J. Weickert.
      "Regularised Diffusion-Shock Inpainting". arXiv preprint. 
      DOI:10.48550/arXiv.2309.08761.
"""

import taichi as ti
import numpy as np
from tqdm import tqdm
from dsfilter.SE2.LI.switches import (
    DS_switch,
    morphological_switch
)
from dsfilter.SE2.LI.derivatives import (
    laplacian,
    morphological
)
from dsfilter.SE2.regularisers import gaussian_derivative_kernel

def DS_filter(u0_np, mask_np, θs_np, T, G_D_inv_np, G_S_inv_np, σ_s, σ_o, ρ_s, ρ_o, ν_s, ν_o, λ, ε=0., dxy=1.):
    """
    Perform Diffusion-Shock inpainting in R^2, according to Schaefer and
    Weickert.[1][2]

    Args:
        `u0_np`: np.ndarray initial condition, with shape [Nx, Ny, Nθ].
        `mask_np`: np.ndarray inpainting mask, with shape [Nx, Ny, Nθ], taking
          values 0 and 1. Wherever the value is 1, no inpainting happens.
        `θs_np`: np.ndarray orientation coordinate θ throughout the domain.
        `T`: time that image is evolved under the DS PDE.
        `G_D_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the diffusion.
        `G_S_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the shock.
        `σ_s`: standard deviation of the internal regularisation in the spatial
          direction of the perpendicular laplacian, used for determining whether
          to perform dilation or erosion.
        `σ_o`: standard deviation of the internal regularisation in the
          orientational direction of the perpendicular laplacian.
        `ρ_s`: standard deviation of the external regularisation in the spatial
          direction of the perpendicular laplacian, used for determining whether
          to perform dilation or erosion.
        `ρ_o`: standard deviation of the external regularisation in the
          orientational direction of the perpendicular laplacian, used for
          determining whether to perform dilation or erosion.
        `ν_s`: standard deviation of the regularisation in the spatial direction
          when taking the perpendicular gradient to determine to what degree
          there is local orientation.
        `ν_o`: standard deviation of the regularisation in the orientational
          direction when taking the perpendicular gradient to determine to what
          degree there is local orientation.
        `λ`: contrast parameter used to determine whether to perform diffusion
          or shock based on the degree of local orientation.
        
      Optional:
        `ε`: regularisation parameter for the signum function used to switch
          between dilation and erosion.
        `dxy`: size of pixels in the x- and y-directions. Defaults to 1.

    Returns:
        np.ndarray solution to the DS PDE with initial condition `u0_np` at
        time `T`.
        TEMP: np.ndarray switch between diffusion and shock, and np.ndarray
        switch between dilation and erosion.

    References:
        [1]: K. Schaefer and J. Weickert.
          "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods in
          Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1137/15M1018460.
        [2]: K. Schaefer and J. Weickert.
          "Regularised Diffusion-Shock Inpainting". arXiv preprint. 
          DOI:10.48550/arXiv.2309.08761.
    """
    # Set hyperparameters
    shape = u0_np.shape
    Nx, Ny, Nθ = shape
    dθ = 2 * np.pi / Nθ
    dt = compute_timestep(dxy, dθ, G_D_inv_np, G_S_inv_np)
    n = int(T / dt)
    # We reuse the Gaussian kernels
    k_s_DS, radius_s_DS = gaussian_derivative_kernel(ν_s, 0)
    k_o_DS, radius_o_DS = gaussian_derivative_kernel(ν_o, 0)
    k_s_morph_int, radius_s_morph_int = gaussian_derivative_kernel(σ_s, 0)
    k_o_morph_int, radius_o_morph_int = gaussian_derivative_kernel(σ_o, 0)
    k_s_morph_ext, radius_s_morph_ext = gaussian_derivative_kernel(ρ_s, 0)
    k_o_morph_ext, radius_o_morph_ext = gaussian_derivative_kernel(ρ_o, 0)

    # Initialise TaiChi objects
    θs = ti.field(ti.f32, shape=shape)
    θs.from_numpy(θs_np)
    G_D_inv = ti.Vector(G_D_inv_np, dt=ti.f32)
    G_S_inv = ti.Vector(G_S_inv_np, dt=ti.f32)
    mask = ti.field(dtype=ti.f32, shape=shape)
    mask.from_numpy(mask_np)
    du_dt = ti.field(dtype=ti.f32, shape=shape)

    ## Padded versions for derivatives
    u = ti.field(dtype=ti.f32, shape=shape)
    u.from_numpy(u0_np)
    ### Laplacian
    laplacian_u = ti.field(dtype=ti.f32, shape=shape)
    ### Morphological
    dilation_u = ti.field(dtype=ti.f32, shape=shape)
    erosion_u = ti.field(dtype=ti.f32, shape=shape)

    ## Fields for switches
    u_switch = ti.field(dtype=ti.f32, shape=shape)
    fill_u_switch(u, u_switch)
    convolution_storage_1 = ti.field(dtype=ti.f32, shape=shape)
    convolution_storage_2 = ti.field(dtype=ti.f32, shape=shape)
    ### DS switch
    gradient_perp_u = ti.field(dtype=ti.f32, shape=shape)
    switch_DS = ti.field(dtype=ti.f32, shape=shape)
    ### Morphological switch
    laplace_perp_u = ti.field(dtype=ti.f32, shape=shape)
    switch_morph = ti.field(dtype=ti.f32, shape=shape)

    for _ in tqdm(range(n)):
        # Compute switches
        DS_switch(u_switch, dxy, θs, k_s_DS, radius_s_DS, k_o_DS, radius_o_DS, λ, gradient_perp_u, switch_DS,
                  convolution_storage_1, convolution_storage_2)
        morphological_switch(u_switch, dxy, θs, ε, k_s_morph_int, radius_s_morph_int, k_o_morph_int, radius_o_morph_int,
                             k_s_morph_ext, radius_s_morph_ext, k_o_morph_ext, radius_o_morph_ext, laplace_perp_u,
                             switch_morph, convolution_storage_1, convolution_storage_2)
        # Compute derivatives
        laplacian(u, G_D_inv, dxy, dθ, θs, laplacian_u)
        morphological(u, G_S_inv, dxy, dθ, θs, dilation_u, erosion_u)
        # Step
        step_DS_filter(u, mask, dt, switch_DS, switch_morph, laplacian_u, dilation_u, erosion_u, du_dt)
        # Step
        # Update fields for switches
        fill_u_switch(u, u_switch)
    return u.to_numpy(), switch_DS.to_numpy(), switch_morph.to_numpy()

def compute_timestep(dxy, dθ, G_D_inv, G_S_inv):
    """
    Compute timestep to solve Diffusion-Shock PDE.
    
    Args:
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in θ direction, taking values greater than 0.
        `G_D_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the diffusion.
        `G_S_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the shock.
    
    Returns:
        timestep, taking values greater than 0.
    """
    τ_D = 1 / ((G_D_inv[0] + G_D_inv[1]) / dxy**2 + G_D_inv[2] / dθ**2)
    τ_M = 1 / np.sqrt((G_S_inv[0] + G_S_inv[1]) / dxy**2 + G_S_inv[2] / dθ**2)
    return 0.1 * min(τ_D, τ_M)


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
        `u`: ti.field(dtype=[float], shape=[Nx+2, Ny+2]) which we want to evolve
          with the DS PDE.
        `du_dt`: ti.field(dtype=[float], shape=[Nx, Ny]) change in `u` in a
          single time step, not taking into account the mask.
    """
    for I in ti.grouped(du_dt):
        du_dt[I] = (
            laplacian_u[I] * switch_DS[I] +
            (1 - switch_DS[I]) * (
                # Do erosion when switch_morph = 1.
                erosion_u[I] * (switch_morph[I] > 0.) * ti.abs(switch_morph[I])  +
                # Do dilation when switch_morph = -1.
                dilation_u[I] * (switch_morph[I] < 0.) * ti.abs(switch_morph[I])
            )
        )
        u[I] += dt * du_dt[I] * (1 - mask[I]) # Only change values in the mask.
        

# Fix padding function

@ti.kernel
def fill_u_switch(
    u: ti.template(),
    u_switch: ti.template()
):
    """
    @taichi.kernel

    Update the content of the field used to determine the switch.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) content to fill
          `padded_u`.
      Mutated:
        `u_switch`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) storage array,
          updated in place.
    """
    for I in ti.grouped(u_switch):
        u_switch[I] = u[I]