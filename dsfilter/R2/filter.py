"""
    filter
    ======

    Provides methods to apply R^2 Diffusion-Shock inpainting, as described by
    K. Schaefer and J. Weickert.[1][2] The primary method is:
      1. `DS_filter`: apply R^2 Diffusion-Shock inpainting to an array
      describing an image, given an inpainting mask and various PDE parameters.

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
from dsfilter.R2.switches import (
    DS_switch,
    morphological_switch
)
from dsfilter.R2.derivatives import (
    laplacian,
    morphological
)
from dsfilter.R2.regularisers import gaussian_derivative_kernel
from dsfilter.utils import (
    compute_PSNR,
    compute_L2,
    compute_L1
)

def DS_inpainting(u0_np, mask_np, T, σ, ρ, ν, λ, ε=0., dxy=1.):
    """
    Perform Diffusion-Shock inpainting in R^2, according to Schaefer and
    Weickert.[1][2]

    Args:
        `u0_np`: np.ndarray initial condition, with shape [Nx, Ny].
        `mask_np`: np.ndarray inpainting mask, with shape [Nx, Ny], taking
          values 0 and 1. Wherever the value is 1, no inpainting happens.
        `T`: time that image is evolved under the DS PDE.
        `σ`: standard deviation of the internal regularisation of the structure
          tensor, used for determining whether to perform dilation or erosion.
        `ρ`: standard deviation of the external regularisation of the structure
          tensor, used for determining whether to perform dilation or erosion.
        `ν`: standard deviation of the regularisation when taking the gradient
          to determine to what degree there is local orientation.
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
    dt = compute_timestep(dxy)
    n = int(T / dt)
    # We reuse the Gaussian kernels
    k_DS, radius_DS = gaussian_derivative_kernel(ν, 0)
    k_morph_int, radius_morph_int = gaussian_derivative_kernel(σ, 0)
    k_morph_ext, radius_morph_ext = gaussian_derivative_kernel(ρ, 0)

    # Initialise TaiChi objects
    shape = u0_np.shape
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
    convolution_storage = ti.field(dtype=ti.f32, shape=shape)
    ### DS switch
    d_dx_DS = ti.field(dtype=ti.f32, shape=shape)
    d_dy_DS = ti.field(dtype=ti.f32, shape=shape)
    switch_DS = ti.field(dtype=ti.f32, shape=shape)
    ### Morphological switch
    u_σ = ti.field(dtype=ti.f32, shape=shape)
    d_dx_morph = ti.field(dtype=ti.f32, shape=shape)
    d_dy_morph = ti.field(dtype=ti.f32, shape=shape)
    Jρ_storage = ti.field(dtype=ti.f32, shape=shape)
    Jρ11 = ti.field(dtype=ti.f32, shape=shape)
    Jρ12 = ti.field(dtype=ti.f32, shape=shape)
    Jρ22 = ti.field(dtype=ti.f32, shape=shape)
    d_dxx = ti.field(dtype=ti.f32, shape=shape)
    d_dxy = ti.field(dtype=ti.f32, shape=shape)
    d_dyy = ti.field(dtype=ti.f32, shape=shape)
    switch_morph = ti.field(dtype=ti.f32, shape=shape)

    for _ in tqdm(range(n)):
        # Compute switches
        DS_switch(u_switch, dxy, k_DS, radius_DS, λ, d_dx_DS, d_dy_DS, switch_DS, convolution_storage)
        morphological_switch(u_switch, u_σ, dxy, ε, k_morph_int, radius_morph_int, d_dx_morph, d_dy_morph, k_morph_ext,
                             radius_morph_ext, Jρ_storage, Jρ11, Jρ12, Jρ22, d_dxx, d_dxy, d_dyy, switch_morph,
                             convolution_storage)
        # Compute derivatives
        laplacian(u, dxy, laplacian_u)
        morphological(u, dxy, dilation_u, erosion_u)
        # Step
        step_DS(u, mask, dt, switch_DS, switch_morph, laplacian_u, dilation_u, erosion_u, du_dt)
        # Update fields for switches
        fill_u_switch(u, u_switch)
    return u.to_numpy()

def DS_enhancing(u0_np, ground_truth_np, T, σ, ρ, ν, λ, ε=0., dxy=1.):
    """
    Perform Diffusion-Shock inpainting in R^2, according to Schaefer and
    Weickert.[1][2]

    Args:
        `u0_np`: np.ndarray initial condition, with shape [Nx, Ny].
        `mask_np`: np.ndarray inpainting mask, with shape [Nx, Ny], taking
          values 0 and 1. Wherever the value is 1, no inpainting happens.
        `T`: time that image is evolved under the DS PDE.
        `σ`: standard deviation of the internal regularisation of the structure
          tensor, used for determining whether to perform dilation or erosion.
        `ρ`: standard deviation of the external regularisation of the structure
          tensor, used for determining whether to perform dilation or erosion.
        `ν`: standard deviation of the regularisation when taking the gradient
          to determine to what degree there is local orientation.
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
    dt = compute_timestep(dxy)
    n = int(T / dt)
    # We reuse the Gaussian kernels
    k_DS, radius_DS = gaussian_derivative_kernel(ν, 0)
    k_morph_int, radius_morph_int = gaussian_derivative_kernel(σ, 0)
    k_morph_ext, radius_morph_ext = gaussian_derivative_kernel(ρ, 0)

    # Initialise TaiChi objects
    shape = u0_np.shape
    mask = ti.field(dtype=ti.f32, shape=shape)
    mask.from_numpy(np.zeros_like(u0_np))
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
    convolution_storage = ti.field(dtype=ti.f32, shape=shape)
    ### DS switch
    d_dx_DS = ti.field(dtype=ti.f32, shape=shape)
    d_dy_DS = ti.field(dtype=ti.f32, shape=shape)
    switch_DS = ti.field(dtype=ti.f32, shape=shape)
    ### Morphological switch
    u_σ = ti.field(dtype=ti.f32, shape=shape)
    d_dx_morph = ti.field(dtype=ti.f32, shape=shape)
    d_dy_morph = ti.field(dtype=ti.f32, shape=shape)
    Jρ_storage = ti.field(dtype=ti.f32, shape=shape)
    Jρ11 = ti.field(dtype=ti.f32, shape=shape)
    Jρ12 = ti.field(dtype=ti.f32, shape=shape)
    Jρ22 = ti.field(dtype=ti.f32, shape=shape)
    d_dxx = ti.field(dtype=ti.f32, shape=shape)
    d_dxy = ti.field(dtype=ti.f32, shape=shape)
    d_dyy = ti.field(dtype=ti.f32, shape=shape)
    switch_morph = ti.field(dtype=ti.f32, shape=shape)

    ## Image Quality Measures
    max_val = 255. # Images are assumed to take gray values in [0, 255].
    ground_truth = ti.field(dtype=ti.f32, shape=shape)
    ground_truth.from_numpy(ground_truth_np)
    PSNR = [compute_PSNR(u, ground_truth, max_val)]
    L1 = [compute_L1(u, ground_truth)]
    L2 = [compute_L2(u, ground_truth)]

    for _ in tqdm(range(n)):
        # Compute switches
        DS_switch(u_switch, dxy, k_DS, radius_DS, λ, d_dx_DS, d_dy_DS, switch_DS, convolution_storage)
        morphological_switch(u_switch, u_σ, dxy, ε, k_morph_int, radius_morph_int, d_dx_morph, d_dy_morph, k_morph_ext,
                             radius_morph_ext, Jρ_storage, Jρ11, Jρ12, Jρ22, d_dxx, d_dxy, d_dyy, switch_morph,
                             convolution_storage)
        # Compute derivatives
        laplacian(u, dxy, laplacian_u)
        morphological(u, dxy, dilation_u, erosion_u)
        # Step
        step_DS(u, mask, dt, switch_DS, switch_morph, laplacian_u, dilation_u, erosion_u, du_dt)
        # Update fields for switches
        fill_u_switch(u, u_switch)

        PSNR.append(compute_PSNR(u, ground_truth, max_val))
        L2.append(compute_L2(u, ground_truth))
        L1.append(compute_L1(u, ground_truth))
    return u.to_numpy(), np.array(PSNR), np.array(L2), np.array(L1), switch_DS.to_numpy(), switch_morph.to_numpy()

def compute_timestep(dxy, δ=np.sqrt(2)-1):
    """
    Compute timestep to solve Diffusion-Shock PDE,[1][2] such that the scheme
    retains the maximum-minimum principle of the continuous PDE.
    
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
    τ_D = compute_timestep_diffusion(dxy, δ)
    τ_M = compute_timestep_shock(dxy, δ)
    return min(τ_D, τ_M) # See Theorem 1 in [2].

def compute_timestep_diffusion(dxy, δ=np.sqrt(2)-1):
    """
    Compute timestep to solve Diffusion-Shock PDE,[1][2] such that the scheme
    retains the maximum-minimum principle of the continuous PDE.
    
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
    return dxy**2 / (4 - 2 * δ)

def compute_timestep_shock(dxy, δ=np.sqrt(2)-1):
    """
    Compute timestep to solve Diffusion-Shock PDE,[1][2] such that the scheme
    retains the maximum-minimum principle of the continuous PDE.
    
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
    return dxy / (np.sqrt(2) * (1 - δ) + δ)


@ti.kernel
def step_DS(
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

    Perform a single timestep Diffusion-Shock inpainting according to Eq. (12)
    in [1] by K. Schaefer and J. Weickert.

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
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) content to fill
          `u_switch`.
      Mutated:
        `u_switch`: ti.field(dtype=[float], shape=[Nx, Ny]) storage array,
          updated in place.
    """
    for I in ti.grouped(u_switch):
        u_switch[I] = u[I]


# Alternative inpainting algorithms
## Diffusion Inpainting
        
def diffusion_inpainting(u0_np, mask_np, dxy, T):
    """
    Perform Diffusion inpainting in R^2.

    Args:
        `u0_np`: np.ndarray initial condition, with shape [Nx, Ny].
        `mask_np`: np.ndarray inpainting mask, with shape [Nx, Ny], taking
          values 0 and 1. Wherever the value is 1, no inpainting happens.
        `dxy`: size of pixels in the x- and y-directions.
        `T`: time that image is evolved under the diffusion PDE.

    Returns:
        np.ndarray solution to the diffusion PDE with initial condition `u0_np`
        at time `T`.
    """
    dt = compute_timestep_diffusion(dxy)
    n = int(T / dt)
    shape = u0_np.shape
    u = ti.field(dtype=ti.f32, shape=shape)
    u.from_numpy(u0_np)
    mask =ti.field(dtype=ti.f32, shape=shape)
    mask.from_numpy(mask_np)
    laplacian_u = ti.field(dtype=ti.f32, shape=shape)
    for _ in tqdm(range(n)):
        laplacian(u, dxy, laplacian_u)
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
        `mask`: ti.field(dtype=[float], shape=[Nx, Ny]) inpainting mask.
        `dt`: step size, taking values greater than 0.
        `laplacian_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of laplacian of
          `u`, which is updated in place.
      Mutated:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) which we want to evolve
          with the diffusion PDE.
    """
    for I in ti.grouped(laplacian_u):
        u[I] += dt * laplacian_u[I] * (1 - mask[I])