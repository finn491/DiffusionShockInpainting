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
      "Regularised Diffusion-Shock Inpainting". In: Journal of Mathematical
      Imaging and Vision (2024).
      DOI:10.1007/s10851-024-01175-0.
"""

import taichi as ti
from dsfilter.SE2.regularisers import (
    convolve_with_kernel_x_dir,
    convolve_with_kernel_y_dir,
    convolve_with_kernel_θ_dir,
    regularise_anisotropic
)
from dsfilter.SE2.gauge.derivatives import (
    laplace_perp,
    gradient_perp
)
from dsfilter.utils import (
    S_ε,
    g_scalar
)

# Isotropic

# Diffusion-Shock

@ti.kernel
def DS_switch(
    u: ti.template(),
    dxy: ti.f32,
    dθ: ti.f32,
    ξ: ti.f32,
    k_s: ti.template(),
    radius_s: ti.template(),
    k_o: ti.template(),
    radius_o: ti.template(),
    λ: ti.f32,
    B2: ti.template(),
    B3: ti.template(),
    gradient_perp_u: ti.template(),
    switch: ti.template(),
    storage: ti.template()
):
    """
    @taichi.kernel

    Determine to what degree we should perform diffusion or shock, as described
    by K. Schaefer and J. Weickert.[1][2]

    Args:
      Static:
        `u`: ti.field(dtype=ti.f32, shape=[Nx, Ny, Nθ]) current state.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
        `ξ`: stiffness parameter defining the cost of moving one unit in the
          orientatonal direction relative to moving one unit in a spatial
          direction, taking values greater than 0.
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
        `storage`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) arrays to hold
          intermediate results when performing convolutions.

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
    # First regularise internally with Gaussian convolution.
    convolve_with_kernel_x_dir(u, k_s, radius_s, switch)
    convolve_with_kernel_y_dir(switch, k_s, radius_s, storage)
    convolve_with_kernel_θ_dir(storage, k_o, radius_o, switch)
    # Then compute perpendicular gradient, which is a measure for lineness.
    gradient_perp(switch, dxy, dθ, ξ, B2, B3, gradient_perp_u)
    for I in ti.grouped(switch):
        switch[I] = g_scalar(gradient_perp_u[I]**2, λ)
    # # Finally regularise externally with Gaussian convolution.
    # convolve_with_kernel_x_dir(switch, k_s, radius_s, gradient_perp_u)
    # convolve_with_kernel_y_dir(gradient_perp_u, k_s, radius_s, storage)
    # convolve_with_kernel_θ_dir(storage, k_o, radius_o, switch)

# Morphological

@ti.kernel
def morphological_switch(
    u: ti.template(),
    dxy: ti.f32,
    dθ: ti.f32,
    ξ: ti.f32,
    ε: ti.f32,
    k_int_s: ti.template(),
    radius_int_s: ti.template(),
    k_int_o: ti.template(),
    radius_int_o: ti.template(),
    k_ext_s: ti.template(),
    radius_ext_s: ti.template(),
    k_ext_o: ti.template(),
    radius_ext_o: ti.template(),
    B2: ti.template(),
    B3: ti.template(),
    laplace_perp_u: ti.template(),
    switch: ti.template(),
    storage: ti.template()
):
    """
    @taichi.func
    
    Determine whether to perform dilation or erosion, as described by
    K. Schaefer and J. Weickert.[1][2]

    Args:
      Static:
        `u`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) current state.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
        `ξ`: stiffness parameter defining the cost of moving one unit in the
          orientatonal direction relative to moving one unit in a spatial
          direction, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
        `ε`: regularisation parameter for the signum function used to switch
          between dilation and erosion, taking values greater than 0.
        `k_int_s`: ti.field(dtype=[float], shape=2*`radius_s`+1) Gaussian kernel
          used for spatial regularisation.
        `radius_int_s`: radius at which kernel `k_s` is truncated, taking
          integer values greater than 0.
        `k_int_o`: ti.field(dtype=[float], shape=2*`radius_o`+1) Gaussian kernel
          used for orientational regularisation.
        `radius_int_o`: radius at which kernel `k_ext` is truncated, taking
          integer values greater than 0.
        `k_ext_s`: ti.field(dtype=[float], shape=2*`radius_s`+1) Gaussian kernel
          used for spatial regularisation.
        `radius_ext_s`: radius at which kernel `k_s` is truncated, taking
          integer values greater than 0.
        `k_ext_o`: ti.field(dtype=[float], shape=2*`radius_o`+1) Gaussian kernel
          used for orientational regularisation.
        `radius_ext_o`: radius at which kernel `k_ext` is truncated, taking
          integer values greater than 0.
      Mutated:
        `laplace_perp_u`: ti.field(dtype=ti.f32, shape=[Nx, Ny, Nθ])
          perpendicular laplacian of u, which is updated in place.
        `switch`: ti.field(dtype=ti.f32, shape=[Nx, Ny, Nθ]) values that
          determine the degree of dilation or erosion, taking values between -1
          and 1, which is updated in place.
        `storage`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) arrays to hold
          intermediate results when performing convolutions.

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
    # First regularise internally with Gaussian convolution.
    convolve_with_kernel_x_dir(u, k_int_s, radius_int_s, switch)
    convolve_with_kernel_y_dir(switch, k_int_s, radius_int_s, storage)
    convolve_with_kernel_θ_dir(storage, k_int_o, radius_int_o, switch)
    # Then compute perpendicular laplacian, which is a measure for convexity.
    laplace_perp(switch, dxy, dθ, ξ, B2, B3, laplace_perp_u)
    for I in ti.grouped(switch):
        switch[I] = (ε > 0.) * S_ε(laplace_perp_u[I], ε) + (ε == 0.) * ti.math.sign(laplace_perp_u[I])
    # Finally regularise externally with Gaussian convolution.
    convolve_with_kernel_x_dir(switch, k_ext_s, radius_ext_s, laplace_perp_u)
    convolve_with_kernel_y_dir(laplace_perp_u, k_ext_s, radius_ext_s, storage)
    convolve_with_kernel_θ_dir(storage, k_ext_o, radius_ext_o, switch)