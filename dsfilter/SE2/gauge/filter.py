"""
    filter
    ======

    Provides methods to apply SE(2) Diffusion-Shock inpainting using gauge
    frames, inspired by the Diffusion-Shock inpainting on R^2 by K. Schaefer and
    J. Weickert.[1][2]
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
from dsfilter.SE2.gauge.switches import (
    DS_switch,
    morphological_switch,
)
from dsfilter.SE2.gauge.derivatives import (
    laplacian,
    morphological,
    TV
)
from dsfilter.SE2.regularisers import gaussian_kernel
from dsfilter.SE2.utils import project_down
from dsfilter.utils import (
    compute_PSNR,
    compute_L2,
    compute_L1
)

def DS_enhancing(u0_np, ground_truth_np, θs_np, ξ, gauge_frame_static, T, G_D_inv_np, G_S_inv_np, σ, ρ, ν, λ, ε=0.,
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
    dt = compute_timestep(dxy, dθ, G_D_inv_np, G_S_inv_np, ξ)
    n = int(T / dt)

    k_s_DS, radius_s_DS = gaussian_kernel(ν, dxy=dxy)
    k_o_DS, radius_o_DS = gaussian_kernel(ν * ξ, dxy=dθ)
    k_s_morph_int, radius_s_morph_int = gaussian_kernel(σ, dxy=dxy)
    k_o_morph_int, radius_o_morph_int = gaussian_kernel(σ * ξ, dxy=dθ)
    k_s_morph_ext, radius_s_morph_ext = gaussian_kernel(ρ, dxy=dxy)
    k_o_morph_ext, radius_o_morph_ext = gaussian_kernel(ρ * ξ, dxy=dθ)

    # Initialise TaiChi objects
    θs = ti.field(ti.f32, shape=shape)
    θs.from_numpy(θs_np)
    G_D_inv = ti.Vector(G_D_inv_np, dt=ti.f32)
    G_S_inv = ti.Vector(G_S_inv_np, dt=ti.f32)
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

    B1_np, B2_np, B3_np = gauge_frame_static
    B1 = ti.Vector.field(3, dtype=ti.f32, shape=shape)
    B1.from_numpy(B1_np)
    B2 = ti.Vector.field(3, dtype=ti.f32, shape=shape)
    B2.from_numpy(B2_np)
    B3 = ti.Vector.field(3, dtype=ti.f32, shape=shape)
    B3.from_numpy(B3_np)

    ## Image Quality Measures
    max_val = 255. # Images are assumed to take gray values in [0, 255].
    ground_truth = ti.field(dtype=ti.f32, shape=shape[:-1])
    ground_truth.from_numpy(ground_truth_np)
    u_projected = ti.field(dtype=ti.f32, shape=shape[:-1])
    project_down(u, u_projected, 0., max_val, 1.)
    PSNR = [compute_PSNR(u_projected, ground_truth, max_val)]
    L1 = [compute_L1(u_projected, ground_truth)]
    L2 = [compute_L2(u_projected, ground_truth)]

    for _ in tqdm(range(n)):
        # Compute switches
        DS_switch(u_switch, dxy, dθ, ξ, k_s_DS, radius_s_DS, k_o_DS, radius_o_DS, λ, B2, B3, gradient_perp_u, switch_DS,
                  storage)
        morphological_switch(u_switch, dxy, dθ, ξ, ε, k_s_morph_int, radius_s_morph_int, k_o_morph_int,
                             radius_o_morph_int, k_s_morph_ext, radius_s_morph_ext, k_o_morph_ext, radius_o_morph_ext,
                             B2, B3, laplace_perp_u, switch_morph, storage)
        # Compute derivatives
        laplacian(u, G_D_inv, dxy, dθ, ξ, B1, B2, B3, laplacian_u)
        morphological(u, G_S_inv, dxy, dθ, ξ, B1, B2, B3, dilation_u, erosion_u)
        # Step
        step_DS(u, dt, switch_DS, switch_morph, laplacian_u, dilation_u, erosion_u, du_dt)
        # Update fields for switches
        fill_u_switch(u, u_switch)

        project_down(u, u_projected, 0., max_val, 1.)
        PSNR.append(compute_PSNR(u_projected, ground_truth, max_val))
        L2.append(compute_L2(u_projected, ground_truth))
        L1.append(compute_L1(u_projected, ground_truth))
    return u.to_numpy(), np.array(PSNR), np.array(L2), np.array(L1), switch_DS.to_numpy(), switch_morph.to_numpy()

@ti.kernel
def step_DS(
    u: ti.template(),
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
        u[I] += dt * du_dt[I]

def compute_timestep(dxy, dθ, G_D_inv, G_S_inv, ξ):
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
    τ_D = compute_timestep_diffusion(dxy, dθ, G_D_inv, ξ)
    τ_M = compute_timestep_shock(dxy, dθ, G_S_inv, ξ)
    return min(τ_D, τ_M)

def compute_timestep_diffusion(dxy, dθ, G_D_inv, ξ):
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
    h = ξ * dxy
    return 1 / (4 * ((G_D_inv[0] + G_D_inv[1]) / h**2 + G_D_inv[2] / dθ**2))

def compute_timestep_shock(dxy, dθ, G_S_inv, ξ):
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
    h = ξ * dxy
    return 1 / (np.sqrt((G_S_inv[0] + G_S_inv[1]) / h**2 + G_S_inv[2] / dθ**2))
        
# TV-Flow

def TV_enhancing(u0_np_unscaled, ground_truth_np, G_inv_np, ξ, dxy, dθ, gauge_frame_static, σ_s, σ_o, T, dt=None, λ=1.):
    """
    Perform Total Variation (TV) Flow inpainting in SE(2).

    Args:
        `u0_np`: np.ndarray initial condition, with shape [Nx, Ny, Nθ].
        `mask_np`: np.ndarray inpainting mask, with shape [Nx, Ny, Nθ], taking
          values 0 and 1. Wherever the value is 1, no inpainting happens.
        `G_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to gauge frame.
        `dxy`: size of pixels in the x- and y-directions.
        `dθ`: size of pixels in the θ-direction.
        `gauge_frame_static`: Tuple[np.ndarray(shape=(Nx, Ny, Nθ, 3)),] gauge
        vector fields B1, B2, and B3 with respect to the static frame.
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
        dt = compute_timestep_TV(dxy, dθ, G_inv_np, ξ)
    print(dt)
    n = int(T / dt)
    k_s, radius_s = gaussian_kernel(σ_s)
    k_o, radius_o = gaussian_kernel(σ_o)
    u0_np = u0_np_unscaled * λ
    shape = u0_np.shape
    u = ti.field(dtype=ti.f32, shape=shape)
    u.from_numpy(u0_np)
    G_inv = ti.Matrix(G_inv_np, dt=ti.f32)
    B1_u = ti.field(dtype=ti.f32, shape=shape)
    B2_u = ti.field(dtype=ti.f32, shape=shape)
    B3_u = ti.field(dtype=ti.f32, shape=shape)
    grad_norm_u = ti.field(dtype=ti.f32, shape=shape)
    normalised_grad_1 = ti.field(dtype=ti.f32, shape=shape)
    normalised_grad_2 = ti.field(dtype=ti.f32, shape=shape)
    normalised_grad_3 = ti.field(dtype=ti.f32, shape=shape)
    TV_u = ti.field(dtype=ti.f32, shape=shape)
    storage = ti.field(dtype=ti.f32, shape=shape)

    B1_np, B2_np, B3_np = gauge_frame_static
    B1 = ti.Vector.field(3, dtype=ti.f32, shape=shape)
    B1.from_numpy(B1_np)
    B2 = ti.Vector.field(3, dtype=ti.f32, shape=shape)
    B2.from_numpy(B2_np)
    B3 = ti.Vector.field(3, dtype=ti.f32, shape=shape)
    B3.from_numpy(B3_np)

    ## Image Quality Measures
    max_val = 255. # Images are assumed to take gray values in [0, 255].
    ground_truth = ti.field(dtype=ti.f32, shape=shape[:-1])
    ground_truth.from_numpy(ground_truth_np)
    u_projected = ti.field(dtype=ti.f32, shape=shape[:-1])
    project_down(u, u_projected, 0., max_val, λ)
    PSNR = [compute_PSNR(u_projected, ground_truth, max_val)]
    L1 = [compute_L1(u_projected, ground_truth)]
    L2 = [compute_L2(u_projected, ground_truth)]

    for _ in tqdm(range(n)):
        TV(u, G_inv, dxy, dθ, ξ, B1, B2, B3, k_s, radius_s, k_o, radius_o, B1_u, B2_u, B3_u, grad_norm_u,
           normalised_grad_1, normalised_grad_2, normalised_grad_3, TV_u, storage)
        step_TV(u, dt, TV_u)

        project_down(u, u_projected, 0., max_val, λ)
        PSNR.append(compute_PSNR(u_projected, ground_truth, max_val))
        L2.append(compute_L2(u_projected, ground_truth))
        L1.append(compute_L1(u_projected, ground_truth))
    return u.to_numpy() / λ, np.array(PSNR), np.array(L2), np.array(L1)
    
@ti.kernel
def step_TV(
    u: ti.template(),
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
        u[I] += dt * TV_u[I]

def compute_timestep_TV(dxy, dθ, G_inv, ξ):
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
    h = ξ * dxy
    return h**2 * dθ / (2 * ((G_inv[0] + G_inv[1]) * h + G_inv[2] * dθ))

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