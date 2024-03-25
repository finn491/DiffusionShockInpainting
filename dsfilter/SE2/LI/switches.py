"""
    switches
    ========

    Provides the operators to switch between diffusion and shock, and between
    dilation and erosion, as described by K. Schaefer and J. Weickert.[1][2]
    The primary methods are:
      1. `DS_switch`: switches between diffusion and shock. If there is locally
      a clear orientation, more shock is applied, see Eq. (7) in [1].
      2. `morphological_switch`: switches between dilation and erosion. If the
      data is locally convex, erosion is applied, while if the data is locally
      concave, dilation is applied, see Eq. (4) in [1].

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
from dsfilter.SE2.regularisers import (
    convolve_with_kernel_x_dir,
    convolve_with_kernel_y_dir,
    convolve_with_kernel_θ_dir
)
from dsfilter.SE2.LI.derivatives import (
    laplace_perp,
    gradient_perp
)
from dsfilter.utils import (
    S_ε,
    g_scalar
)

# Diffusion-Shock

## Switcher

@ti.kernel
def DS_switch(
    u: ti.template(),
    dxy: ti.f32,
    θs: ti.template(),
    k_s: ti.template(),
    radius_s: ti.i32,
    k_o: ti.template(),
    radius_o: ti.i32,
    λ: ti.f32,
    gradient_perp_u: ti.template(),
    switch: ti.template(),
    convolution_storage_1: ti.template(),
    convolution_storage_2: ti.template()
):
    """
    @taichi.kernel

    Determine to what degree we should perform diffusion or shock, as described
    by K. Schaefer and J. Weickert.[1][2]

    Args:
      Static:
        `u`: ti.field(dtype=ti.f32, shape=[Nx, Ny, Nθ]) current state.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
        `k_s`: ti.field(dtype=ti.f32, shape=2*`radius_s`+1) Gaussian kernel used
          for spatial regularisation.
        `radius_s`: radius at which kernel `k_s` is truncated, taking integer
          values greater than 0.
        `k_o`: ti.field(dtype=ti.f32, shape=2*`radius_o`+1) Gaussian kernel used
          for orientational regularisation.
        `radius_o`: radius at which kernel `k_o` is truncated, taking integer
          values greater than 0.
        `λ`: contrast parameter, taking values greater than 0.
      Mutated:
        `gradient_perp_u`: ti.field(dtype=ti.f32, shape=[Nx, Ny, Nθ])
          perpendicular gradient of u, which is updated in place.
        `switch`: ti.field(dtype=ti.f32, shape=[Nx, Ny, Nθ]) values that
          determine the degree of diffusion or shock, taking values between 0
          and 1, which is updated in place.
        `convolution_storage_*`: ti.field(dtype=[float], shape=[Nx, Ny]) arrays
          to hold intermediate results when performing convolutions.

    References:
        [1]: K. Schaefer and J. Weickert.
          "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods in
          Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1137/15M1018460.
        [2]: K. Schaefer and J. Weickert.
          "Regularised Diffusion-Shock Inpainting". arXiv preprint. 
          DOI:10.48550/arXiv.2309.08761.
    """
    # First regularise with Gaussian convolution.
    convolve_with_kernel_x_dir(u, k_s, radius_s, convolution_storage_1)
    convolve_with_kernel_y_dir(convolution_storage_1, k_s, radius_s, convolution_storage_2)
    convolve_with_kernel_θ_dir(convolution_storage_2, k_o, radius_o, switch)
    # Then compute perpendicular gradient, which is a measure for lineness.
    gradient_perp(switch, dxy, θs, gradient_perp_u)
    for I in ti.grouped(switch):
        switch[I] = g_scalar(gradient_perp_u[I]**2, λ)

# Morphological

## Switcher

@ti.kernel
def morphological_switch(
    u: ti.template(),
    dxy: ti.f32,
    θs: ti.template(),
    ε: ti.f32,
    k_s: ti.template(),
    radius_s: ti.i32,
    k_o: ti.template(),
    radius_o: ti.i32,
    laplace_perp_u: ti.template(),
    switch: ti.template(),
    convolution_storage_1: ti.template(),
    convolution_storage_2: ti.template()
):
    """
    @taichi.func
    
    Determine whether to perform dilation or erosion, as described by
    K. Schaefer and J. Weickert.[1][2]

    Args:
      Static:
        `u`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) current state.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
        `ε`: regularisation parameter for the signum function used to switch
          between dilation and erosion, taking values greater than 0.
        `k_s`: ti.field(dtype=[float], shape=2*`radius_s`+1) Gaussian kernel
          used for spatial regularisation.
        `radius_s`: radius at which kernel `k_s` is truncated, taking
          integer values greater than 0.
        `k_o`: ti.field(dtype=[float], shape=2*`radius_o`+1) Gaussian kernel
          used for orientational regularisation.
        `radius_o`: radius at which kernel `k_ext` is truncated, taking
          integer values greater than 0.
      Mutated:
        `laplace_perp_u`: ti.field(dtype=ti.f32, shape=[Nx, Ny, Nθ])
          perpendicular laplacian of u, which is updated in place.
        `switch`: ti.field(dtype=ti.f32, shape=[Nx, Ny, Nθ]) values that
          determine the degree of dilation or erosion, taking values between -1
          and 1, which is updated in place.
        `convolution_storage_*`: ti.field(dtype=[float], shape=[Nx, Ny]) arrays
          to hold intermediate results when performing convolutions.

    References:
        [1]: K. Schaefer and J. Weickert.
          "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods in
          Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1137/15M1018460.
        [2]: K. Schaefer and J. Weickert.
          "Regularised Diffusion-Shock Inpainting". arXiv preprint. 
          DOI:10.48550/arXiv.2309.08761.
    """
    # First regularise with Gaussian convolution.
    convolve_with_kernel_x_dir(u, k_s, radius_s, convolution_storage_1)
    convolve_with_kernel_y_dir(convolution_storage_1, k_s, radius_s, convolution_storage_2)
    convolve_with_kernel_θ_dir(convolution_storage_2, k_o, radius_o, switch)
    # Then compute perpendicular gradient, which is a measure for lineness.
    laplace_perp(switch, dxy, θs, laplace_perp_u)
    for I in ti.grouped(switch):
        switch[I] = (ε > 0.) * S_ε(laplace_perp_u[I], ε) + (ε == 0.) * ti.math.sign(laplace_perp_u[I])