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
from dsfilter.R3.switches import (
    DS_switch,
    morphological_switch
)
from dsfilter.R3.derivatives import (
    laplacian,
    morphological
)
from dsfilter.R3.regularisers import gaussian_kernel
from dsfilter.utils import unpad_array

def DS_filter(u0_np, mask_np, T, σ, ρ, ν, λ, ε=0., dxy=1., dz = 1.):
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
    factor = 0.05
    dt = 0.05
    n = int(T / dt)
    # We reuse the Gaussian kernels
    k_DS, radius_DS = gaussian_kernel(ν)
    k_morph_int, radius_morph_int = gaussian_kernel(σ)
    k_morph_ext, radius_morph_ext = gaussian_kernel(ρ)

    k_DSz, radius_DSz = gaussian_kernel(ν*factor, dxy = dz)
    k_morph_intz, radius_morph_intz = gaussian_kernel(σ*factor, dxy = dz)
    k_morph_extz, radius_morph_extz = gaussian_kernel(ρ*factor, dxy = dz)

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
    convolution_storage2 = ti.field(dtype=ti.f32, shape=shape)
    ### DS switch
    d_dx_DS = ti.field(dtype=ti.f32, shape=shape)
    d_dy_DS = ti.field(dtype=ti.f32, shape=shape)
    d_dz_DS = ti.field(dtype=ti.f32, shape=shape)
    switch_DS = ti.field(dtype=ti.f32, shape=shape)
    ### Morphological switch
    u_σ = ti.field(dtype=ti.f32, shape=shape)
    d_dx_morph = ti.field(dtype=ti.f32, shape=shape)
    d_dy_morph = ti.field(dtype=ti.f32, shape=shape)
    d_dz_morph = ti.field(dtype=ti.f32, shape=shape)
    d_dxx = ti.field(dtype=ti.f32, shape=shape)
    d_dxy = ti.field(dtype=ti.f32, shape=shape)
    d_dyy = ti.field(dtype=ti.f32, shape=shape)
    d_dxz = ti.field(dtype=ti.f32, shape=shape)
    d_dyz = ti.field(dtype=ti.f32, shape=shape)
    d_dzz = ti.field(dtype=ti.f32, shape=shape)
    dxdx = ti.field(dtype=ti.f32, shape=shape)
    dxdy = ti.field(dtype=ti.f32, shape=shape)
    dydy = ti.field(dtype=ti.f32, shape=shape)
    dxdz = ti.field(dtype=ti.f32, shape=shape)
    dydz = ti.field(dtype=ti.f32, shape=shape)
    dzdz = ti.field(dtype=ti.f32, shape=shape)

    switch_morph = ti.field(dtype=ti.f32, shape=shape)

    for _ in tqdm(range(n)):
        # Compute switches
        DS_switch(u_switch, dxy, dz, k_DS, radius_DS, k_DSz, radius_DSz, λ, d_dx_DS, d_dy_DS, d_dz_DS, switch_DS, convolution_storage, convolution_storage2)
        morphological_switch(u_switch, u_σ, dxy, dz, ε, k_morph_int, radius_morph_int, k_morph_intz, radius_morph_intz, d_dx_morph, d_dy_morph, d_dz_morph, k_morph_ext,
                             radius_morph_ext, k_morph_extz, radius_morph_extz, d_dxx, d_dxy, d_dyy, d_dxz, d_dyz, d_dzz, switch_morph,
                             convolution_storage, convolution_storage2, dxdx, dxdy, dydy,dxdz,dydz,dzdz)
        # Compute derivatives
        laplacian(u, dxy, dz, laplacian_u)
        morphological(u, dxy, dz,  dilation_u, erosion_u)
        # Step
        step_DS_filter(u, mask, dt, switch_DS, switch_morph, laplacian_u, dilation_u, erosion_u, du_dt)
        # Update fields for switches
        fill_u_switch(u, u_switch)
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
        assert switch_DS[I] <=1, 'assert ds'
        assert switch_morph[I] <=1, 'assert ds1'
        assert switch_morph[I] >=-1, 'assert ds2'
        du_dt[I] = (
            laplacian_u[I] * switch_DS[I] +
            (1 - switch_DS[I]) * (
                # Do erosion when switch_morph = 1.
                erosion_u[I] * (switch_morph[I] > 0.)   +
                # Do dilation when switch_morph = -1.
                dilation_u[I] * (switch_morph[I] < 0.) 
            ) 
        )
        if mask[I] < 1:
          u[I] += dt * du_dt[I] # Only change values in the mask.
        

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