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
import numpy
import taichi as ti
from dsfilter.R3.regularisers import (
    convolve_with_kernel_x_dir,
    convolve_with_kernel_y_dir,
    convolve_with_kernel_z_dir
)
from dsfilter.R3.derivatives import central_derivatives_first_order, central_derivatives_second_order
from dsfilter.R3.utils import sanitize_index
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
    k: ti.template(),
    radius: ti.i32,
    λ: ti.f32,
    d_dx: ti.template(),
    d_dy: ti.template(),
    d_dz: ti.template(),
    switch: ti.template(),
    convolution_storage: ti.template(),
    convolution_storage2: ti.template()
):
    """
    @taichi.kernel

    Determine to what degree we should perform diffusion or shock, as described
    by K. Schaefer and J. Weickert.[1][2]

    Args:
      Static:
        `u`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) current state.
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) Gaussian kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
        `λ`: contrast parameter, taking values greater than 0.
      Mutated:
        `d_d*`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of derivatives, which are
          updated in place.
        `switch`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) values that
          determine the degree of diffusion or shock, taking values between 0
          and 1, which is updated in place.

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
    # First regularise with Gaussian convolution.
    convolve_with_kernel_x_dir(u, k, radius, convolution_storage)
    convolve_with_kernel_y_dir(convolution_storage, k, radius, convolution_storage2)
    convolve_with_kernel_z_dir(convolution_storage2, k, radius, switch)
    # Then compute gradient with Sobel operators.
    I_dx = ti.Vector([1, 0, 0], dt=ti.i32)
    I_dy = ti.Vector([0, 1, 0], dt=ti.i32)
    I_dz = ti.Vector([0, 0, 1], dt=ti.i32)
    for I in ti.grouped(switch):
        I_dx_forward = sanitize_index(I + I_dx, u)
        I_dx_backward = sanitize_index(I - I_dx, u)
        I_dy_forward = sanitize_index(I + I_dy, u)
        I_dy_backward = sanitize_index(I - I_dy, u)
        I_dz_forward = sanitize_index(I + I_dz, u)
        I_dz_backward = sanitize_index(I - I_dz, u) 
        d_dx[I] = 0.5 * (u[I_dx_forward] - u[I_dx_backward]) / dxy
        d_dy[I] = 0.5 * (u[I_dy_forward] - u[I_dy_backward]) / dxy
        d_dz[I] = 0.5 * (u[I_dz_forward] - u[I_dz_backward]) / dxy
        switch[I] = g_scalar(d_dx[I]**2 + d_dy[I]**2 + d_dz[I]**2, λ)

# Morphological

## Switcher

@ti.kernel
def morphological_switch(
    u: ti.template(),
    u_σ: ti.template(),
    dxy: ti.f32,
    ε: ti.f32,
    k_int: ti.template(),
    radius_int: ti.i32,
    d_dx: ti.template(),
    d_dy: ti.template(),
    d_dz: ti.template(),
    k_ext: ti.template(),
    radius_ext: ti.template(),
    d_dxx: ti.template(),
    d_dxy: ti.template(),
    d_dyy: ti.template(),
    d_dxz: ti.template(),
    d_dyz: ti.template(),
    d_dzz: ti.template(),
    switch: ti.template(),
    convolution_storage: ti.template(),
    convolution_storage2: ti.template(),
    dxdx: ti.template(),
    dxdy: ti.template(),
    dydy: ti.template(),
    dxdz: ti.template(),
    dydz: ti.template(),
    dzdz: ti.template()
):
    """
    @taichi.func
    
    Determine whether to perform dilation or erosion, as described by
    K. Schaefer and J. Weickert.[1][2]

    Args:
      Static:
        `u`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) current state.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `ε`: regularisation parameter for the signum function used to switch
          between dilation and erosion, taking values greater than 0.
        `k_int`: ti.field(dtype=[float], shape=2*`radius_int`+1) Gaussian kernel
          with standard deviation σ.
        `radius_int`: radius at which kernel `k_int` is truncated, taking
          integer values greater than 0.
        `k_ext`: ti.field(dtype=[float], shape=2*`radius_ext`+1) Gaussian kernel
          with standard deviation ρ.
        `radius_ext`: radius at which kernel `k_ext` is truncated, taking
          integer values greater than 0.
      Mutated:
        `u_σ`: ti.field(dtype=[float], shape=[Nx, Ny]) u convolved with Gaussian
          with standard deviation σ.
        `d_d*`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of first order Gaussian
          derivatives, which are updated in place.
        `Jρ_storage`: ti.field(dtype=[float], shape=[Nx, Ny]) array to hold
          intermediate computations for the structure tensor.
        `Jρ**`: ti.field(dtype=[float], shape=[Nx, Ny]) **-component of the
          regularised structure tensor.
        `d_d**`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of second order Gaussian
          derivatives, which are updated in place.
        `switch`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) values that
          determine the degree of dilation or erosion, taking values between -1
          and 1, which is updated in place.
        `convolution_storage`: ti.field(dtype=[float], shape=[Nx, Ny]) array to
          hold intermediate results when performing convolutions.

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
    #presmooth image 
    convolve_with_kernel_x_dir(u, k_int, radius_int, convolution_storage)
    convolve_with_kernel_y_dir(convolution_storage, k_int, radius_int, convolution_storage2)  
    convolve_with_kernel_z_dir(convolution_storage2, k_int, radius_int, switch)        
    # compute entries of the structure tensor

    central_derivatives_first_order(u, dxy, d_dx, d_dy, d_dz)       

    # Compute second derivative of u_σ in the direction of the dominant
    # eigenvector.
    central_derivatives_second_order(switch, dxy, d_dxx, d_dxy, d_dyy, d_dxz, d_dyz, d_dzz)

    for I in ti.grouped(switch):  
        # a = numpy.ones(switch.shape)
        # a = numpy.array([[d_dx[I]**2      , d_dy[I]*d_dx[I],  d_dz[I]*d_dx[I]],
        #                  [d_dy[I]*d_dx[I] , d_dy[I]**2,       d_dz[I]*d_dy[I]],
        #                  [d_dz[I]*d_dx[I] , d_dz[I]*d_dy[I],  d_dz[I]**2]])
        dxdx[I] = d_dx[I]**2
        dydy[I] = d_dy[I]**2
        dzdz[I] = d_dz[I]**2
        dxdy[I] = d_dx[I]*d_dy[I]
        dydz[I] = d_dz[I]*d_dy[I]
        dxdz[I] = d_dx[I]*d_dz[I]

    convolve_with_kernel_x_dir(dxdx, k_ext, radius_ext, convolution_storage)
    convolve_with_kernel_y_dir(convolution_storage, k_ext, radius_ext, convolution_storage2)  
    convolve_with_kernel_z_dir(convolution_storage2, k_ext, radius_ext, dxdx) 
    convolve_with_kernel_x_dir(dydy, k_ext, radius_ext, convolution_storage)
    convolve_with_kernel_y_dir(convolution_storage, k_ext, radius_ext, convolution_storage2)  
    convolve_with_kernel_z_dir(convolution_storage2, k_ext, radius_ext, dydy) 
    convolve_with_kernel_x_dir(dzdz, k_ext, radius_ext, convolution_storage)
    convolve_with_kernel_y_dir(convolution_storage, k_ext, radius_ext, convolution_storage2)  
    convolve_with_kernel_z_dir(convolution_storage2, k_ext, radius_ext, dzdz) 
    convolve_with_kernel_x_dir(dxdy, k_ext, radius_ext, convolution_storage)
    convolve_with_kernel_y_dir(convolution_storage, k_ext, radius_ext, convolution_storage2)  
    convolve_with_kernel_z_dir(convolution_storage2, k_ext, radius_ext, dxdy) 
    convolve_with_kernel_x_dir(dydz, k_ext, radius_ext, convolution_storage)
    convolve_with_kernel_y_dir(convolution_storage, k_ext, radius_ext, convolution_storage2)  
    convolve_with_kernel_z_dir(convolution_storage2, k_ext, radius_ext, dydz) 
    convolve_with_kernel_x_dir(dxdz, k_ext, radius_ext, convolution_storage)
    convolve_with_kernel_y_dir(convolution_storage, k_ext, radius_ext, convolution_storage2)  
    convolve_with_kernel_z_dir(convolution_storage2, k_ext, radius_ext, dxdz) 

    for I in ti.grouped(switch):  
      a = ti.Matrix([[dxdx[I], dxdy[I], dxdz[I] ],
                     [dxdy[I], dydy[I], dydz[I] ],
                     [dxdz[I], dydz[I], dzdz[I] ] ])
      values, v = ti.sym_eig(a)
      i = 2
      if values[0] > values[1] and values[0] > values[2]:
          i = 0
      else:
          if values[1] > values[0] and values[1] > values[2]:
            i = 1

      d_dww = v[0,i]**2 * d_dxx[I] + v[1,i]**2 *  d_dyy[I] + v[2,i]**2 *  d_dzz[I] +  2* v[0,i]* v[1,i] * d_dxy[I] +  2* v[0,i]* v[2,i] * d_dxz[I] +  2* v[1,i] * v[2,i] * d_dyz[I]
      switch[I] = (ε > 0.) * S_ε(d_dww, ε) + (ε == 0.) * ti.math.sign(d_dww)
    
      



#done