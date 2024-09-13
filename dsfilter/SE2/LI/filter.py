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
      "Regularised Diffusion-Shock Inpainting". In: Journal of Mathematical
      Imaging and Vision (2024).
      DOI:10.1007/s10851-024-01175-0.
"""

import taichi as ti
import numpy as np
from tqdm import tqdm
from dsfilter.SE2.LI.switches import (
    DS_switch,
    DS_switch_s,
    DS_switch_o,
    morphological_switch,
    morphological_switch_s,
    morphological_switch_o
)
from dsfilter.SE2.LI.derivatives import (
    laplacian,
    laplacian_s,
    laplacian_o,
    morphological,
    morphological_s,
    morphological_o,
    TV
)
from dsfilter.SE2.regularisers import gaussian_derivative_kernel

def DS_filter(u0_np, mask_np, θs_np, ξ, T, G_D_inv_np, G_S_inv_np, σ, ρ, ν, λ, ε=0.,
              dxy=1.):
    """
    Perform Diffusion-Shock inpainting in SE(2), using an adaptation of the 
    R^2 Diffusion-Shock inpainting algorithm described by Schaefer and
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
        `σ_*`: standard deviation in the A*-direction of the internal
          regularisation, taking values greater than 0.
        `ρ_*`: standard deviation in the A*-direction of the external
          regularisation, taking values greater than 0.
        `ν_*`: standard deviation in the A*-direction of the internal and
          external regularisation, taking values greater than 0.
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
          "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods
          in Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1137/15M1018460.
        [2]: K. Schaefer and J. Weickert.
          "Regularised Diffusion-Shock Inpainting". In: Journal of Mathematical
          Imaging and Vision (2024).
          DOI:10.1007/s10851-024-01175-0.
    """
    # Set hyperparameters
    shape = u0_np.shape
    _, _, Nθ = shape
    dθ = 2 * np.pi / Nθ
    dt = compute_timestep(dxy, dθ, G_D_inv_np, G_S_inv_np)
    n = int(T / dt)

    k_s_DS, radius_s_DS = gaussian_derivative_kernel(ν, 0, dxy=dxy)
    k_o_DS, radius_o_DS = gaussian_derivative_kernel(ν * ξ, 0, dxy=dθ)
    k_s_morph_int, radius_s_morph_int = gaussian_derivative_kernel(σ, 0, dxy=dxy)
    k_o_morph_int, radius_o_morph_int = gaussian_derivative_kernel(σ * ξ, 0, dxy=dθ)
    k_s_morph_ext, radius_s_morph_ext = gaussian_derivative_kernel(ρ, 0, dxy=dxy)
    k_o_morph_ext, radius_o_morph_ext = gaussian_derivative_kernel(ρ * ξ, 0, dxy=dθ)

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
    storage = ti.field(dtype=ti.f32, shape=shape)
    ### DS switch
    gradient_u = ti.field(dtype=ti.f32, shape=shape)
    switch_DS = ti.field(dtype=ti.f32, shape=shape)
    ### Morphological switch
    laplace_perp_u = ti.field(dtype=ti.f32, shape=shape)
    switch_morph = ti.field(dtype=ti.f32, shape=shape)

    for _ in tqdm(range(n)):
        # Compute switches
        DS_switch(u_switch, dxy, dθ, ξ, θs, k_s_DS, radius_s_DS, k_o_DS, radius_o_DS, λ, gradient_u, switch_DS, storage)
        morphological_switch(u_switch, dxy, dθ, ξ, θs, ε, k_s_morph_int, radius_s_morph_int, k_o_morph_int,
                             radius_o_morph_int, k_s_morph_ext, radius_s_morph_ext, k_o_morph_ext, radius_o_morph_ext,
                             laplace_perp_u, switch_morph, storage)
        # Compute derivatives
        laplacian(u, G_D_inv, dxy, dθ, θs, laplacian_u)
        morphological(u, G_S_inv, dxy, dθ, θs, dilation_u, erosion_u)
        # Step
        step_DS_filter(u, mask, dt, switch_DS, switch_morph, laplacian_u, dilation_u, erosion_u, du_dt)
        # Update fields for switches
        fill_u_switch(u, u_switch)
    # ti.sync()
    # ti.profiler.print_kernel_profiler_info("trace")
    # ti.profiler.clear_kernel_profiler_info()
    return u.to_numpy(), switch_DS.to_numpy(), switch_morph.to_numpy()

# def DS_filter_spatial(u0_np, mask_np, θs_np, T, G_D_inv_np, G_S_inv_np, σ_s, σ_o, ρ_s, ρ_o, ν_s, ν_o, λ, ε=0., dxy=1.):
#     """
#     Perform Diffusion-Shock inpainting in SE(2), using an adaptation of the 
#     R^2 Diffusion-Shock inpainting algorithm described by Schaefer and
#     Weickert.[1][2]

#     Args:
#         `u0_np`: np.ndarray initial condition, with shape [Nx, Ny, Nθ].
#         `mask_np`: np.ndarray inpainting mask, with shape [Nx, Ny, Nθ], taking
#           values 0 and 1. Wherever the value is 1, no inpainting happens.
#         `θs_np`: np.ndarray orientation coordinate θ throughout the domain.
#         `T`: time that image is evolved under the DS PDE.
#         `G_D_inv_np`: np.ndarray(shape=(2,), dtype=[float]) spatial constants of
#           the inverse of the diagonal metric tensor with respect to left
#           invariant basis used to define the diffusion.
#         `G_S_inv_np`: np.ndarray(shape=(2,), dtype=[float]) spatial constants of
#           the inverse of the diagonal metric tensor with respect to left
#           invariantbasis used to define the shock.
#         `σ_s`: standard deviation in the spatial directions of the internal
#           regularisation of the morphological switch, taking values greater than
#           0.
#         `σ_o`: standard deviation in the orientational directions of the
#           internal regularisation of the morphological switch, taking values
#           greater than 0.
#         `ρ_s`: standard deviation in the spatial directions of the external
#           regularisation of the morphological switch, taking values greater than
#           0.
#         `ρ_o`: standard deviation in the orientational directions of the
#           external regularisation of the morphological switch, taking values
#           greater than 0.
#         `ν_s`: standard deviation in the spatial directions of the internal
#           regularisation of the morphological switch, taking values greater than
#           0.
#         `ν_o`: standard deviation in the orientational directions of the
#           internal regularisation of the diffusion-shock switch, taking values
#           greater than 0.
#         `λ`: contrast parameter used to determine whether to perform diffusion
#           or shock based on the degree of local orientation.
        
#       Optional:
#         `ε`: regularisation parameter for the signum function used to switch
#           between dilation and erosion.
#         `dxy`: size of pixels in the x- and y-directions. Defaults to 1.

#     Returns:
#         np.ndarray solution to the DS PDE with initial condition `u0_np` at
#         time `T`.
#         TEMP: np.ndarray switch between diffusion and shock, and np.ndarray
#         switch between dilation and erosion.

#     References:
#         [1]: K. Schaefer and J. Weickert.
#           "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods
#           in Computer Vision 14009 (2023), pp. 588--600.
#           DOI:10.1137/15M1018460.
#         [2]: K. Schaefer and J. Weickert.
#           "Regularised Diffusion-Shock Inpainting". In: Journal of Mathematical
#           Imaging and Vision (2024).
#           DOI:10.1007/s10851-024-01175-0.
#     """
#     # Set hyperparameters
#     shape = u0_np.shape
#     _, _, Nθ = shape
#     dθ = 2 * np.pi / Nθ
#     dt = compute_timestep(dxy, dθ, G_D_inv_np, G_S_inv_np)
#     n = int(T / dt)

#     k_s_DS, radius_s_DS = gaussian_derivative_kernel(ν_s, 0, dxy=dxy)
#     k_o_DS, radius_o_DS = gaussian_derivative_kernel(ν_o, 0, dxy=dθ)
#     k_s_morph_int, radius_s_morph_int = gaussian_derivative_kernel(σ_s, 0, dxy=dxy)
#     k_o_morph_int, radius_o_morph_int = gaussian_derivative_kernel(σ_o, 0, dxy=dθ)
#     k_s_morph_ext, radius_s_morph_ext = gaussian_derivative_kernel(ρ_s, 0, dxy=dxy)
#     k_o_morph_ext, radius_o_morph_ext = gaussian_derivative_kernel(ρ_o, 0, dxy=dθ)

#     # Initialise TaiChi objects
#     θs = ti.field(ti.f32, shape=shape)
#     θs.from_numpy(θs_np)
#     G_D_inv = ti.Vector(G_D_inv_np[:-1], dt=ti.f32)
#     G_S_inv = ti.Vector(G_S_inv_np[:-1], dt=ti.f32)
#     mask = ti.field(dtype=ti.f32, shape=shape)
#     mask.from_numpy(mask_np)
#     du_dt = ti.field(dtype=ti.f32, shape=shape)

#     ## Padded versions for derivatives
#     u = ti.field(dtype=ti.f32, shape=shape)
#     u.from_numpy(u0_np)
#     ### Laplacian
#     laplacian_u = ti.field(dtype=ti.f32, shape=shape)
#     ### Morphological
#     dilation_u = ti.field(dtype=ti.f32, shape=shape)
#     erosion_u = ti.field(dtype=ti.f32, shape=shape)

#     ## Fields for switches
#     u_switch = ti.field(dtype=ti.f32, shape=shape)
#     fill_u_switch(u, u_switch)
#     storage = ti.field(dtype=ti.f32, shape=shape)
#     ### DS switch
#     gradient_perp_u = ti.field(dtype=ti.f32, shape=shape)
#     switch_DS = ti.field(dtype=ti.f32, shape=shape)
#     ### Morphological switch
#     laplace_perp_u = ti.field(dtype=ti.f32, shape=shape)
#     switch_morph = ti.field(dtype=ti.f32, shape=shape)

#     for _ in tqdm(range(n)):
#         # Compute switches
#         DS_switch_s(u_switch, dxy, θs, k_s_DS, radius_s_DS, k_o_DS, radius_o_DS, λ, gradient_perp_u, switch_DS, storage)
#         morphological_switch_s(u_switch, dxy, θs, ε, k_s_morph_int, radius_s_morph_int, k_o_morph_int,
#                                radius_o_morph_int, k_s_morph_ext, radius_s_morph_ext, k_o_morph_ext, radius_o_morph_ext,
#                                laplace_perp_u, switch_morph, storage)
#         # Compute derivatives
#         laplacian_s(u, G_D_inv, dxy, θs, laplacian_u)
#         morphological_s(u, G_S_inv, dxy, θs, dilation_u, erosion_u)
#         # Step
#         step_DS_filter(u, mask, dt, switch_DS, switch_morph, laplacian_u, dilation_u, erosion_u, du_dt)
#         # Update fields for switches
#         fill_u_switch(u, u_switch)
#     # ti.sync()
#     # ti.profiler.print_kernel_profiler_info("trace")
#     # ti.profiler.clear_kernel_profiler_info()
#     return u.to_numpy(), switch_DS.to_numpy(), switch_morph.to_numpy()

def DS_filter_spatial(u0_np, mask_np, θs_np, T, G_D_inv_np, G_S_inv_np, σ_s, σ_o, ρ_s, ρ_o, ν_s, ν_o, λ, ε=0., dxy=1.):
    """
    Perform Diffusion-Shock inpainting in SE(2), using an adaptation of the 
    R^2 Diffusion-Shock inpainting algorithm described by Schaefer and
    Weickert.[1][2]

    Args:
        `u0_np`: np.ndarray initial condition, with shape [Nx, Ny, Nθ].
        `mask_np`: np.ndarray inpainting mask, with shape [Nx, Ny, Nθ], taking
          values 0 and 1. Wherever the value is 1, no inpainting happens.
        `θs_np`: np.ndarray orientation coordinate θ throughout the domain.
        `T`: time that image is evolved under the DS PDE.
        `G_D_inv_np`: np.ndarray(shape=(2,), dtype=[float]) spatial constants of
          the inverse of the diagonal metric tensor with respect to left
          invariant basis used to define the diffusion.
        `G_S_inv_np`: np.ndarray(shape=(2,), dtype=[float]) spatial constants of
          the inverse of the diagonal metric tensor with respect to left
          invariantbasis used to define the shock.
        `σ_s`: standard deviation in the spatial directions of the internal
          regularisation of the morphological switch, taking values greater than
          0.
        `σ_o`: standard deviation in the orientational directions of the
          internal regularisation of the morphological switch, taking values
          greater than 0.
        `ρ_s`: standard deviation in the spatial directions of the external
          regularisation of the morphological switch, taking values greater than
          0.
        `ρ_o`: standard deviation in the orientational directions of the
          external regularisation of the morphological switch, taking values
          greater than 0.
        `ν_s`: standard deviation in the spatial directions of the internal
          regularisation of the morphological switch, taking values greater than
          0.
        `ν_o`: standard deviation in the orientational directions of the
          internal regularisation of the diffusion-shock switch, taking values
          greater than 0.
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
          "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods
          in Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1137/15M1018460.
        [2]: K. Schaefer and J. Weickert.
          "Regularised Diffusion-Shock Inpainting". In: Journal of Mathematical
          Imaging and Vision (2024).
          DOI:10.1007/s10851-024-01175-0.
    """
    # Set hyperparameters
    shape = u0_np.shape
    _, _, Nθ = shape
    dθ = 2 * np.pi / Nθ
    dt = compute_timestep(dxy, dθ, G_D_inv_np, G_S_inv_np)
    n = int(T / dt)

    k_s_DS, radius_s_DS = gaussian_derivative_kernel(ν_s, 0, dxy=dxy)
    k_o_DS, radius_o_DS = gaussian_derivative_kernel(ν_o, 0, dxy=dθ)
    k_s_morph_int, radius_s_morph_int = gaussian_derivative_kernel(σ_s, 0, dxy=dxy)
    k_o_morph_int, radius_o_morph_int = gaussian_derivative_kernel(σ_o, 0, dxy=dθ)
    k_s_morph_ext, radius_s_morph_ext = gaussian_derivative_kernel(ρ_s, 0, dxy=dxy)
    k_o_morph_ext, radius_o_morph_ext = gaussian_derivative_kernel(ρ_o, 0, dxy=dθ)

    # Initialise TaiChi objects
    θs = ti.field(ti.f32, shape=shape)
    θs.from_numpy(θs_np)
    G_D_inv = ti.Vector(G_D_inv_np[:-1], dt=ti.f32)
    G_S_inv = ti.Vector(G_S_inv_np[:-1], dt=ti.f32)
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
    storage = ti.field(dtype=ti.f32, shape=shape)
    ### DS switch
    gradient_u = ti.field(dtype=ti.f32, shape=shape)
    switch_DS = ti.field(dtype=ti.f32, shape=shape)
    ### Morphological switch
    laplace_perp_u = ti.field(dtype=ti.f32, shape=shape)
    switch_morph = ti.field(dtype=ti.f32, shape=shape)

    for _ in tqdm(range(n)):
        # Compute switches
        DS_switch_s(u_switch, dxy, θs, k_s_DS, radius_s_DS, k_o_DS, radius_o_DS, λ, gradient_u, switch_DS, storage)
        morphological_switch_s(u_switch, dxy, θs, ε, k_s_morph_int, radius_s_morph_int, k_o_morph_int,
                               radius_o_morph_int, k_s_morph_ext, radius_s_morph_ext, k_o_morph_ext, radius_o_morph_ext,
                               laplace_perp_u, switch_morph, storage)
        # Compute derivatives
        laplacian_s(u, G_D_inv, dxy, θs, laplacian_u)
        morphological_s(u, G_S_inv, dxy, θs, dilation_u, erosion_u)
        # Step
        step_DS_filter(u, mask, dt, switch_DS, switch_morph, laplacian_u, dilation_u, erosion_u, du_dt)
        # Update fields for switches
        fill_u_switch(u, u_switch)
    # ti.sync()
    # ti.profiler.print_kernel_profiler_info("trace")
    # ti.profiler.clear_kernel_profiler_info()
    return u.to_numpy(), switch_DS.to_numpy(), switch_morph.to_numpy()

def DS_filter_mixed(u0_np, mask_np, θs_np, T, G_D_inv_np, G_S_inv_np, σ_s, σ_o, ρ_s, ρ_o, ν_s, ν_o, λ_s, λ_o, ε_s=0.,
                    ε_o=0., dxy=1.):
    """
    Perform Diffusion-Shock inpainting in SE(2), using an adaptation of the 
    R^2 Diffusion-Shock inpainting algorithm described by Schaefer and
    Weickert.[1][2]

    Args:
        `u0_np`: np.ndarray initial condition, with shape [Nx, Ny, Nθ].
        `mask_np`: np.ndarray inpainting mask, with shape [Nx, Ny, Nθ], taking
          values 0 and 1. Wherever the value is 1, no inpainting happens.
        `θs_np`: np.ndarray orientation coordinate θ throughout the domain.
        `T`: time that image is evolved under the DS PDE.
        `G_D_inv_np`: np.ndarray(shape=(3,), dtype=[float]) constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the diffusion.
        `G_S_inv_np`: np.ndarray(shape=(3,), dtype=[float]) constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the shock.
        `σ_s`: standard deviation in the spatial directions of the internal
          regularisation of the morphological switch, taking values greater than
          0.
        `σ_o`: standard deviation in the orientational directions of the
          internal regularisation of the morphological switch, taking values
          greater than 0.
        `ρ_s`: standard deviation in the spatial directions of the external
          regularisation of the morphological switch, taking values greater than
          0.
        `ρ_o`: standard deviation in the orientational directions of the
          external regularisation of the morphological switch, taking values
          greater than 0.
        `ν_s`: standard deviation in the spatial directions of the internal
          regularisation of the morphological switch, taking values greater than
          0.
        `ν_o`: standard deviation in the orientational directions of the
          internal regularisation of the diffusion-shock switch, taking values
          greater than 0.
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
          "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods
          in Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1137/15M1018460.
        [2]: K. Schaefer and J. Weickert.
          "Regularised Diffusion-Shock Inpainting". In: Journal of Mathematical
          Imaging and Vision (2024).
          DOI:10.1007/s10851-024-01175-0.
    """
    # Set hyperparameters
    shape = u0_np.shape
    _, _, Nθ = shape
    dθ = 2 * np.pi / Nθ
    dt = compute_timestep(dxy, dθ, G_D_inv_np, G_S_inv_np)
    n = int(T / dt)

    k_s_DS, radius_s_DS = gaussian_derivative_kernel(ν_s, 0, dxy=dxy)
    k_o_DS, radius_o_DS = gaussian_derivative_kernel(ν_o, 0, dxy=dθ)
    k_s_morph_int, radius_s_morph_int = gaussian_derivative_kernel(σ_s, 0, dxy=dxy)
    k_o_morph_int, radius_o_morph_int = gaussian_derivative_kernel(σ_o, 0, dxy=dθ)
    k_s_morph_ext, radius_s_morph_ext = gaussian_derivative_kernel(ρ_s, 0, dxy=dxy)
    k_o_morph_ext, radius_o_morph_ext = gaussian_derivative_kernel(ρ_o, 0, dxy=dθ)

    # Initialise TaiChi objects
    θs = ti.field(ti.f32, shape=shape)
    θs.from_numpy(θs_np)
    G_D_inv_s = ti.Vector(G_D_inv_np[:-1], dt=ti.f32)
    G_S_inv_s = ti.Vector(G_S_inv_np[:-1], dt=ti.f32)
    G_D_inv_o = G_D_inv_np[-1]
    G_S_inv_o = G_S_inv_np[-1]
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
    storage = ti.field(dtype=ti.f32, shape=shape)
    ### DS switch
    gradient_perp_u = ti.field(dtype=ti.f32, shape=shape)
    switch_DS = ti.field(dtype=ti.f32, shape=shape)
    ### Morphological switch
    laplace_perp_u = ti.field(dtype=ti.f32, shape=shape)
    switch_morph = ti.field(dtype=ti.f32, shape=shape)

    for _ in tqdm(range(n)):
        # Spatial update
        # Compute switches
        DS_switch_s(u_switch, dxy, θs, k_s_DS, radius_s_DS, k_o_DS, radius_o_DS, λ_s, gradient_perp_u, switch_DS,
                    storage)
        morphological_switch_s(u_switch, dxy, θs, ε_s, k_s_morph_int, radius_s_morph_int, k_o_morph_int,
                               radius_o_morph_int, k_s_morph_ext, radius_s_morph_ext, k_o_morph_ext, radius_o_morph_ext,
                               laplace_perp_u, switch_morph, storage)
        # Compute derivatives
        laplacian_s(u, G_D_inv_s, dxy, θs, laplacian_u)
        morphological_s(u, G_S_inv_s, dxy, θs, dilation_u, erosion_u)
        # Step
        step_DS_filter(u, mask, dt, switch_DS, switch_morph, laplacian_u, dilation_u, erosion_u, du_dt)
        # Update fields for switches
        fill_u_switch(u, u_switch)
        # Orientational update
        # Compute switches
        DS_switch_o(u_switch, dθ, k_s_DS, radius_s_DS, k_o_DS, radius_o_DS, λ_o, gradient_perp_u, switch_DS, storage)
        morphological_switch_o(u_switch, dθ, ε_o, k_s_morph_int, radius_s_morph_int, k_o_morph_int, radius_o_morph_int,
                               k_s_morph_ext, radius_s_morph_ext, k_o_morph_ext, radius_o_morph_ext, laplace_perp_u,
                               switch_morph, storage)
        # Compute derivatives
        laplacian_o(u, G_D_inv_o, dθ, laplacian_u)
        morphological_o(u, G_S_inv_o, dθ, dilation_u, erosion_u)
        # Step
        step_DS_filter(u, mask, dt, switch_DS, switch_morph, laplacian_u, dilation_u, erosion_u, du_dt)
        # Update fields for switches
        fill_u_switch(u, u_switch)
    # ti.sync()
    # ti.profiler.print_kernel_profiler_info("trace")
    # ti.profiler.clear_kernel_profiler_info()
    return u.to_numpy(), switch_DS.to_numpy(), switch_morph.to_numpy()

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

    Perform a single timestep of SE(2) Diffusion-Shock inpainting, adaptating
    the R^2 Diffusion-Shock inpainting algorithm described by Schaefer and
    Weickert.[1][2]

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

    References:
        [1]: K. Schaefer and J. Weickert.
          "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods
          in Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1137/15M1018460.
        [2]: K. Schaefer and J. Weickert.
          "Regularised Diffusion-Shock Inpainting". In: Journal of Mathematical
          Imaging and Vision (2024).
          DOI:10.1007/s10851-024-01175-0.
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
    τ_D = compute_timestep_diffusion(dxy, dθ, G_D_inv)
    τ_M = compute_timestep_shock(dxy, dθ, G_S_inv)
    return min(τ_D, τ_M)

def compute_timestep_diffusion(dxy, dθ, G_D_inv):
    """
    Compute timestep to solve Diffusion PDE.
    
    Args:
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in θ direction, taking values greater than 0.
        `G_D_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the diffusion.
    
    Returns:
        timestep, taking values greater than 0.
    """
    return 1 / (4 * ((G_D_inv[0] + G_D_inv[1]) / dxy**2 + G_D_inv[2] / dθ**2))

def compute_timestep_shock(dxy, dθ, G_S_inv):
    """
    Compute timestep to solve Shock PDE.
    
    Args:
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in θ direction, taking values greater than 0.
        `G_S_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the shock.
    
    Returns:
        timestep, taking values greater than 0.
    """
    return 1 / (np.sqrt((G_S_inv[0] + G_S_inv[1]) / dxy**2 + G_S_inv[2] / dθ**2))
        
# Alternative inpainting algorithms
## Diffusion Inpainting
        
def diffusion_inpainting(u0_np, mask_np, G_inv_np, dxy, dθ, θs_np, T):
    """
    Perform Diffusion inpainting in SE(2).

    Args:
        `u0_np`: np.ndarray initial condition, with shape [Nx, Ny, Nθ].
        `mask_np`: np.ndarray inpainting mask, with shape [Nx, Ny, Nθ], taking
          values 0 and 1. Wherever the value is 1, no inpainting happens.
        `G_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the diffusion.
        `dxy`: size of pixels in the x- and y-directions.
        `dθ`: size of pixels in the θ-direction.
        `θs_np`: np.ndarray orientation coordinate θ throughout the domain.
        `T`: time that image is evolved under the diffusion PDE.

    Returns:
        np.ndarray solution to the diffusion PDE with initial condition `u0_np`
        at time `T`.
    """
    dt = compute_timestep_diffusion(dxy, dθ, G_inv_np)
    n = int(T / dt)
    shape = u0_np.shape
    u = ti.field(dtype=ti.f32, shape=shape)
    u.from_numpy(u0_np)
    mask =ti.field(dtype=ti.f32, shape=shape)
    mask.from_numpy(mask_np)
    G_inv = ti.Matrix(G_inv_np, dt=ti.f32)
    θs = ti.field(dtype=ti.f32, shape=shape)
    θs.from_numpy(θs_np)
    laplacian_u = ti.field(dtype=ti.f32, shape=shape)
    for _ in tqdm(range(n)):
        laplacian(u, G_inv, dxy, dθ, θs, laplacian_u)
        step_diffusion_inpainting(u, mask, dt, laplacian_u)
    return u.to_numpy()
    
@ti.kernel
def step_diffusion_inpainting(
    u: ti.template(),
    mask: ti.template(),
    dt: ti.f32,
    laplacian_u: ti.template()
):
    """
    @taichi.kernel

    Perform a single timestep of diffusion inpainting.

    Args:
      Static:
        `mask`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) inpainting mask.
        `dt`: step size, taking values greater than 0.
        `laplacian_u`: ti.field(dtype=[float], shape=[Nx, Ny, N θ]) of laplacian
          of `u`, which is updated in place.
      Mutated:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to evolve
          with the diffusion PDE.
    """
    for I in ti.grouped(laplacian_u):
        u[I] += dt * laplacian_u[I] * (1 - mask[I])

## Shock Inpainting

def shock_inpainting(u0_np, mask_np, G_inv_np, dxy, dθ, θs_np, σ_s, σ_o, ρ_s, ρ_o, ε, T):
    """
    Perform Shock inpainting in SE(2).

    Args:
        `u0_np`: np.ndarray initial condition, with shape [Nx, Ny, Nθ].
        `mask_np`: np.ndarray inpainting mask, with shape [Nx, Ny, Nθ], taking
          values 0 and 1. Wherever the value is 1, no inpainting happens.
        `θs_np`: np.ndarray orientation coordinate θ throughout the domain.
        `G_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the shock.
        `dxy`: size of pixels in the x- and y-directions.
        `dθ`: size of pixels in the θ-direction.
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
        `ε`: regularisation parameter for the signum function used to switch
          between dilation and erosion.
        `T`: time that image is evolved under the DS PDE.

    Returns:
        np.ndarray solution to the DS PDE with initial condition `u0_np` at
        time `T`.
    """
    dt = compute_timestep_shock(dxy, dθ, G_inv_np)
    n = int(T / dt)
    k_s_int, radius_s_int = gaussian_derivative_kernel(σ_s, 0)
    k_o_int, radius_o_int = gaussian_derivative_kernel(σ_o, 0)
    k_s_ext, radius_s_ext = gaussian_derivative_kernel(ρ_s, 0)
    k_o_ext, radius_o_ext = gaussian_derivative_kernel(ρ_o, 0)
    shape = u0_np.shape
    u = ti.field(dtype=ti.f32, shape=shape)
    u.from_numpy(u0_np)
    mask =ti.field(dtype=ti.f32, shape=shape)
    mask.from_numpy(mask_np)
    G_inv = ti.Matrix(G_inv_np, dt=ti.f32)
    θs = ti.field(dtype=ti.f32, shape=shape)
    θs.from_numpy(θs_np)
    dilation_U = ti.field(dtype=ti.f32, shape=shape)
    erosion_U = ti.field(dtype=ti.f32, shape=shape)
    laplace_perp_U = ti.field(dtype=ti.f32, shape=shape)
    switch = ti.field(dtype=ti.f32, shape=shape)
    convolution_storage_1 = ti.field(dtype=ti.f32, shape=shape)
    convolution_storage_2 = ti.field(dtype=ti.f32, shape=shape)

    # ξ = 1.

    for _ in tqdm(range(n)):
        morphological_switch(u, dxy, θs, ε, k_s_int, radius_s_int, k_o_int, radius_o_int, k_s_ext, radius_s_ext, # , dθ, ξ
                             k_o_ext, radius_o_ext, laplace_perp_U, switch, convolution_storage_1,
                             convolution_storage_2)
        morphological(u, G_inv, dxy, dθ, θs, dilation_U, erosion_U)
        step_shock_inpainting(u, mask, dt, switch, dilation_U, erosion_U)
    return u.to_numpy()
    
@ti.kernel
def step_shock_inpainting(
    u: ti.template(),
    mask: ti.template(),
    dt: ti.f32,
    switch: ti.template(),
    dilation_U: ti.template(),
    erosion_U: ti.template()
):
    """
    @taichi.kernel

    Perform a single timestep of SE(2) Shock inpainting.

    Args:
      Static:
        `mask`: ti.field(dtype=[float], shape=[Nx, Ny]) inpainting mask.
        `dt`: step size, taking values greater than 0.
        `switch`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of values that
          determine the degree of dilation or erosion, taking values between -1
          and 1.
        `dilation_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of ||grad `u`||,
          which is updated in place.
        `erosion_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of -||grad `u`||,
          which is updated in place.
      Mutated:
        `u`: ti.field(dtype=[float], shape=[Nx+2, Ny+2]) which we want to evolve
          with the shock PDE.
    """
    for I in ti.grouped(dilation_U):
        u[I] += dt * (
                # Do erosion when switch_morph = 1.
                erosion_U[I] * (switch[I] > 0.) * ti.abs(switch[I])  +
                # Do dilation when switch_morph = -1.
                dilation_U[I] * (switch[I] < 0.) * ti.abs(switch[I])
            ) * (1 - mask[I])

# TV-Flow

def TV_inpainting(u0_np, mask_np, G_inv_np, dxy, dθ, θs_np, σ_s, σ_o, T, dt=None):
    """
    Perform Total Variation (TV) Flow inpainting in SE(2).

    Args:
        `u0_np`: np.ndarray initial condition, with shape [Nx, Ny, Nθ].
        `mask_np`: np.ndarray inpainting mask, with shape [Nx, Ny, Nθ], taking
          values 0 and 1. Wherever the value is 1, no inpainting happens.
        `θs_np`: np.ndarray orientation coordinate θ throughout the domain.
        `G_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis.
        `dxy`: size of pixels in the x- and y-directions.
        `dθ`: size of pixels in the θ-direction.
        `σ_s`: standard deviation of the internal regularisation in the spatial
          direction of the perpendicular laplacian, used for determining whether
          to perform dilation or erosion.
        `σ_o`: standard deviation of the internal regularisation in the
          orientational direction of the perpendicular laplacian.
        `T`: time that image is evolved under the DS PDE.

    Returns:
        np.ndarray solution to the DS PDE with initial condition `u0_np` at
        time `T`.
    """
    if dt is None:
        dt = compute_timestep_TV(dxy, dθ, G_inv_np)
    print(dt)
    n = int(T / dt)
    k_s, radius_s = gaussian_derivative_kernel(σ_s, 0)
    k_o, radius_o = gaussian_derivative_kernel(σ_o, 0)
    shape = u0_np.shape
    u = ti.field(dtype=ti.f32, shape=shape)
    u.from_numpy(u0_np)
    mask =ti.field(dtype=ti.f32, shape=shape)
    mask.from_numpy(mask_np)
    G_inv = ti.Matrix(G_inv_np, dt=ti.f32)
    θs = ti.field(dtype=ti.f32, shape=shape)
    θs.from_numpy(θs_np)
    A1_u = ti.field(dtype=ti.f32, shape=shape)
    A2_u = ti.field(dtype=ti.f32, shape=shape)
    A3_u = ti.field(dtype=ti.f32, shape=shape)
    grad_norm_u = ti.field(dtype=ti.f32, shape=shape)
    normalised_grad_1 = ti.field(dtype=ti.f32, shape=shape)
    normalised_grad_2 = ti.field(dtype=ti.f32, shape=shape)
    normalised_grad_3 = ti.field(dtype=ti.f32, shape=shape)
    TV_u = ti.field(dtype=ti.f32, shape=shape)
    storage = ti.field(dtype=ti.f32, shape=shape)

    for _ in tqdm(range(n)):
        TV(u, G_inv, dxy, dθ, θs, k_s, radius_s, k_o, radius_o, A1_u, A2_u, A3_u, grad_norm_u, normalised_grad_1,
           normalised_grad_2, normalised_grad_3, TV_u, storage)
        step_TV_inpainting(u, mask, dt, TV_u)
    return u.to_numpy()
    
@ti.kernel
def step_TV_inpainting(
    u: ti.template(),
    mask: ti.template(),
    dt: ti.f32,
    TV_u: ti.template(),
):
    """
    @taichi.kernel

    Perform a single timestep of SE(2) Shock inpainting.

    Args:
      Static:
        `mask`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) inpainting mask.
        `dt`: step size, taking values greater than 0.
        `TV_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of 
          div(grad `u` / ||grad `u`||), which is updated in place.
      Mutated:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to evolve
          with the shock PDE.
    """
    for I in ti.grouped(TV_u):
        u[I] += dt * TV_u[I] * (1 - mask[I])

def compute_timestep_TV(dxy, dθ, G_inv):
    """
    Compute timestep to solve TV flow.
    
    Args:
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in θ direction, taking values greater than 0.
        `G_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the TV flow.
    
    Returns:
        timestep, taking values greater than 0.
    """
    return dxy**2 * dθ / (2 * ((G_inv[0] + G_inv[1]) * dxy + G_inv[2] * dθ))

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