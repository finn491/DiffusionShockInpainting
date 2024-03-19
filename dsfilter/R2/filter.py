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

def DS_filter_R2(u0_np, mask_np, T, σ, ρ, ν, λ, ε=0., dxy=1.):
    """
    Perform Diffusion-Shock inpainting in R^2, according to Schaefer and
    Weickert "Diffusion-Shock Inpainting" (2023).

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
    """
    # Set hyperparameters
    dt = compute_timestep(dxy)
    n = int(T / dt)
    # We reuse the Gaussian kernels
    k_DS, radius_DS = gaussian_derivative_kernel(ν, 0)
    k_morph_int, radius_morph_int = gaussian_derivative_kernel(σ, 0)
    k_morph_ext, radius_morph_ext = gaussian_derivative_kernel(ρ, 0)

    # Initialise TaiChi objects
    Nx, Ny = u0_np.shape
    u0 = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    u0.from_numpy(u0_np)
    mask = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    mask.from_numpy(mask_np)
    du_dt = ti.field(dtype=ti.f32, shape=(Nx, Ny))

    ## Padded versions for derivatives
    u = ti.field(dtype=ti.f32, shape=(Nx + 2, Ny + 2))
    fill_padded_u(u0, 2, u)
    ### Laplacian
    laplacian_u = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    ### Morphological
    dilation_u = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    erosion_u = ti.field(dtype=ti.f32, shape=(Nx, Ny))

    ## Padded versions for switches
    ### DS switch
    u_DS = ti.field(dtype=ti.f32, shape=(Nx + 2 * radius_DS, Ny + 2 * radius_DS))
    fill_padded_u(u, radius_DS, u_DS)
    d_dx_DS = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    d_dy_DS = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    switch_DS = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    ### Morphological switch
    u_structure_tensor = ti.field(dtype=ti.f32, shape=(Nx + 2 * (radius_morph_int + radius_morph_ext),
                                                       Ny + 2 * (radius_morph_int + radius_morph_ext)))
    fill_padded_u(u, radius_morph_ext + radius_morph_int, u_structure_tensor)
    u_σ_structure_tensor = ti.field(dtype=ti.f32, shape=(Nx + 2 * radius_morph_ext, Ny + 2 * radius_morph_ext))
    d_dx_morph = ti.field(dtype=ti.f32, shape=(Nx + 2 * radius_morph_ext, Ny + 2 * radius_morph_ext))
    d_dy_morph = ti.field(dtype=ti.f32, shape=(Nx + 2 * radius_morph_ext, Ny + 2 * radius_morph_ext))
    Jρ_padded = ti.field(dtype=ti.f32, shape=(Nx + 2 * radius_morph_ext, Ny + 2 * radius_morph_ext))
    Jρ11 = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    Jρ12 = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    Jρ22 = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    u_dominant_derivative = ti.field(dtype=ti.f32, shape=(Nx + 2 * radius_morph_int, Ny + 2 * radius_morph_int))
    fill_padded_u(u, radius_morph_int, u_dominant_derivative)
    d_dxx = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    d_dxy = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    d_dyy = ti.field(dtype=ti.f32, shape=(Nx, Ny))
    switch_morph = ti.field(dtype=ti.f32, shape=(Nx, Ny))

    for _ in tqdm(range(n)):
        # Compute switches
        DS_switch(u_DS, dxy, k_DS, radius_DS, λ, d_dx_DS, d_dy_DS, switch_DS)
        morphological_switch(u_structure_tensor, u_σ_structure_tensor, u_dominant_derivative, dxy, ε, k_morph_int,
                             radius_morph_int, d_dx_morph, d_dy_morph, k_morph_ext, radius_morph_ext, Jρ_padded, Jρ11,
                             Jρ12, Jρ22, d_dxx, d_dxy, d_dyy, switch_morph)
        # Compute derivatives
        laplacian(u, dxy, laplacian_u)
        morphological(u, dxy, dilation_u, erosion_u)
        # Step
        step_DS_filter(u, mask, dt, switch_DS, switch_morph, laplacian_u, dilation_u, erosion_u, du_dt)
        # Correct boundary conditions
        ## For derivatives
        fix_reflected_padding(u, 1)
        ## For switches
        ### DS switch
        fill_padded_u(u, radius_DS, u_DS)
        fix_reflected_padding(u_DS, radius_DS)
        ### Morphological switch
        fill_padded_u(u, radius_morph_ext + radius_morph_int, u_structure_tensor)
        fix_reflected_padding(u_structure_tensor, radius_morph_ext + radius_morph_int)
        fill_padded_u(u, radius_morph_int, u_dominant_derivative)
        fix_reflected_padding(u_dominant_derivative, radius_morph_ext)
    return unpad_array(u.to_numpy(), pad_shape=1), switch_DS.to_numpy(), switch_morph.to_numpy()

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
    return min(τ_D, τ_M) #/ 2


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
    I_shift = ti.Vector([1, 1], dt=ti.i32)
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
        u[I + I_shift] += dt * du_dt[I] * (1 - mask[I]) # Only change values in the mask.
        

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
            u[i, k] = u[i, 2 * radius - k] # u[i, radius]
            u[i, J-1 - k] = u[i, J-1 - 2 * radius + k] # u[i, J-1 - radius]
    for j in range(J):
        for k in range(radius):
            u[k, j] = u[2 * radius - k, j] # u[radius, j]
            u[I-1 - k, j] = u[I-1 - 2 * radius + k, j] # u[I-1 - radius, j]

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