# switches.py

import taichi as ti
from dsfilter.R2.derivatives import (
    convolve_with_kernel_x_dir,
    convolve_with_kernel_y_dir,
    central_derivatives_second_order
)
from dsfilter.R2.utils import sanitize_index

# Diffusion-Shock

## Switcher

@ti.kernel
def DS_switch(
    u_padded: ti.template(),
    dxy: ti.f32,
    k: ti.template(),
    radius: ti.i32,
    λ: ti.f32,
    d_dx: ti.template(),
    d_dy: ti.template(),
    switch: ti.template()
):
    """
    @taichi.kernel

    Determine to what degree we should perform diffusion or shock, as in
    "Diffusion-Shock Inpainting" (2023) by K. Schaefer and J. Weickert.

    Args:
      Static:
        `u_padded`: ti.field(dtype=ti.f32, shape=[Nx+2*`radius`, Ny+2*`radius`])
          array to be convolved.
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) Gaussian kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
        `λ`: contrast parameter, taking values greater than 0.
      Mutated:
        `d_d*`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of derivatives, which are
          updated in place.
        `switch`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of values that
          determine the degree of diffusion or shock, taking values between 0
          and 1, which is updated in place.
    """
    # First regularise with Gaussian convolution.
    convolve_with_kernel_x_dir(u_padded, k, radius, switch)
    convolve_with_kernel_y_dir(u_padded, k, radius, switch)
    # Then compute gradient with Sobel operators.
    sobel_gradient(switch, dxy, d_dx, d_dy)
    for I in ti.grouped(switch):
        switch[I] = g_scalar(d_dx[I]**2 + d_dy[I]**2, λ)

@ti.func
def g_scalar(
    s_squared: ti.f32, 
    λ: ti.f32
) -> ti.f32:
    """
    @taichi.func
    
    Compute g, the function that switches between diffusion and shock in
    "Diffusion-Shock Inpainting" (2023) by K. Schaefer and J. Weickert.

    Args:
        `s_squared`: square of some scalar, taking values greater than 0.
        `λ`: contrast parameter, taking values greater than 0.

    Returns:
        ti.f32 of g(`s_squared`).
    """
    return 1 / ti.math.sqrt(1 + s_squared / λ**2)


# Morphological

## Switcher

@ti.kernel
def morphological_switch(
    u_padded_ext: ti.template(),
    u_σ_padded: ti.template(),
    u_padded_int: ti.template(),
    dxy: ti.f32,
    k_int: ti.template(),
    radius_int: ti.i32,
    d_dx: ti.template(),
    d_dy: ti.template(),
    k_ext: ti.template(),
    radius_ext: ti.template(),
    Jρ_padded: ti.template(),
    Jρ11: ti.template(),
    Jρ12: ti.template(),
    Jρ22: ti.template(),
    c: ti.template(),
    s: ti.template(),
    d_dxx: ti.template(),
    d_dxy: ti.template(),
    d_dyy: ti.template(),
    switch: ti.template()
):
    """
    @taichi.func
    
    Determine whether to perform dilation or erosion, as in
    "Diffusion-Shock Inpainting" (2023) by K. Schaefer and J. Weickert.

    Args:
      Static:
        `u_padded`: ti.field(dtype=ti.f32, shape=[Nx+2*`radius`, Ny+2*`radius`])
          padded current state.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `k_int`: ti.field(dtype=[float], shape=2*`radius_int`+1) Gaussian kernel
          with standard deviation σ.
        `radius_int`: radius at which kernel `k_int` is truncated, taking
          integer values greater than 0.
        `k_ext`: ti.field(dtype=[float], shape=2*`radius_ext`+1) Gaussian kernel
          with standard deviation ρ.
        `radius_ext`: radius at which kernel `k_ext` is truncated, taking
          integer values greater than 0.
        `u`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of current state.
      Mutated:
        `u_σ_padded`: ti.field(dtype=[float], shape=[Nx+2*`radius_ext`, Ny+2*`radius_ext`])
          padded u convolved with Gaussian with standard deviation σ.
        `d_d*`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of first order Gaussian
          derivatives, which are updated in place.
        `Jρ_padded`: ti.field(dtype=[float], shape=[Nx+2*`radius_ext`, Ny+2*`radius_ext`])
          padded array to hold intermediate computations for the structure
          tensor.
        `Jρ**`: ti.field(dtype=[float], shape=[Nx, Ny]) **-component of the
          regularised structure tensor.
        `c`: ti.field(dtype=[float], shape=[Nx, Ny]) first components of the
          normalised dominant eigenvectors, as in Eq. (15).
        `s`: ti.field(dtype=[float], shape=[Nx, Ny]) second components of the
          normalised dominant eigenvectors, as in Eq. (15).
        `d_d**`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of second order Gaussian
          derivatives, which are updated in place.
        `switch`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of values that
          determine the degree of dilation or erosion, taking values between -1
          and 1, which is updated in place.
    """
    # Find dominant eigenvector of structure tensor.
    find_dominant_eigenvector(u_padded_ext, u_σ_padded, dxy, k_int, radius_int, d_dx, d_dy, k_ext, radius_ext,
                              Jρ_padded, Jρ11, Jρ12, Jρ22, c, s)
    # Regularise with same Gaussian kernel as when computing gradient for
    # structure tensor.
    convolve_with_kernel_x_dir(u_padded_int, k_int, radius_int, switch)
    convolve_with_kernel_y_dir(u_padded_int, k_int, radius_int, switch)
    # Compute second derivatives of u_σ.
    central_derivatives_second_order(switch, dxy, k_int, radius_int, d_dxx, d_dxy, d_dyy)
    # Compute second derivative of u_σ in the direction of the dominant
    # eigenvector.
    for I in ti.grouped(switch):
        switch[I] = ti.math.sign(c[I]**2 * d_dxx[I] + 2 * c[I] * s[I] * d_dxy[I] + s[I]**2 * d_dyy[I])

@ti.func
def find_dominant_eigenvector(
    u_padded: ti.template(),
    u_σ_padded: ti.template(),
    dxy: ti.f32,
    k_int: ti.template(),
    radius_int: ti.i32,
    d_dx: ti.template(),
    d_dy: ti.template(),
    k_ext: ti.template(),
    radius_ext: ti.template(),
    Jρ_padded: ti.template(),
    Jρ11: ti.template(),
    Jρ12: ti.template(),
    Jρ22: ti.template(),
    c: ti.template(),
    s: ti.template()
):
    """
    Compute the dominant eigenvector of the structure tensor, as in
    "Diffusion-Shock Inpainting" (2023) by K. Schaefer and J. Weickert.

    Args:
      Static:
        `u_padded`: ti.field(dtype=[float], shape=[Nx+2*(`radius_ext`+`radius_int`), Ny+2*(`radius_ext`+`radius_int`)]),
          padded array of which the dominant eigenvectors of the structure
          tensor are to be found.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `k_int`: ti.field(dtype=[float], shape=2*`radius_int`+1) Gaussian kernel
          with standard deviation σ.
        `radius_int`: radius at which kernel `k_int` is truncated, taking
          integer values greater than 0.
        `k_ext`: ti.field(dtype=[float], shape=2*`radius_ext`+1) Gaussian kernel
          with standard deviation ρ.
        `radius_ext`: radius at which kernel `k_ext` is truncated, taking
          integer values greater than 0.
      Mutated:
        `u_σ_padded`: ti.field(dtype=[float], shape=[Nx+2*`radius_ext`, Ny+2*`radius_ext`])
          padded u convolved with Gaussian with standard deviation σ.
        `d_d*`: ti.field(dtype=[float], shape=[Nx+2*`radius_ext`, Ny+2*`radius_ext`])
          Gaussian derivatives, which are updated in place.
        `Jρ_padded`: ti.field(dtype=[float], shape=[Nx+2*`radius_ext`, Ny+2*`radius_ext`])
          padded array to hold intermediate computations for the structure
          tensor.
        `Jρ**`: ti.field(dtype=[float], shape=[Nx, Ny]) **-component of the
          regularised structure tensor.
        `c`: ti.field(dtype=[float], shape=[Nx, Ny]) first components of the
          normalised dominant eigenvectors, as in Eq. (15).
        `s`: ti.field(dtype=[float], shape=[Nx, Ny]) second components of the
          normalised dominant eigenvectors, as in Eq. (15).
    """
    # First regularise with Gaussian convolution.
    convolve_with_kernel_x_dir(u_padded, k_int, radius_ext + radius_int, u_σ_padded)
    convolve_with_kernel_y_dir(u_padded, k_int, radius_ext + radius_int, u_σ_padded)
    # Then compute gradient with Sobel operators.
    sobel_gradient(u_σ_padded, dxy, d_dx, d_dy)
    # Compute components of structure tensor.
    compute_structure_tensor(d_dx, d_dy, k_ext, radius_ext, Jρ_padded, Jρ11, Jρ12, Jρ22)
    # Compute dominant eigenvector.
    for I in ti.grouped(c):
        A11 = Jρ11[I]
        A12 = Jρ12[I]
        A22 = Jρ22[I]
        # The dominant eigenvector of a symmetrix 2x2 matrix A with nonnegative
        # trace A11 + A22, such as the structure tensor, is given by
        #   (-(-A11 + A22 - sqrt((A11 - A22)**2 + 4 A12**2))/(2 A12), 1).
        v1 = -(-A11 + A22 - ti.math.sqrt((A11 - A22)**2 + 4 * A12**2)) / (2 * A12)
        norm = ti.math.sqrt(v1**2 + 1)
        c[I] = v1 / norm
        s[I] = 1 / norm

@ti.func
def compute_structure_tensor(
    d_dx: ti.template(),
    d_dy: ti.template(),
    k_ext: ti.template(),
    radius_ext: ti.i32,
    Jρ_padded: ti.template(),
    Jρ11: ti.template(),
    Jρ12: ti.template(),
    Jρ22: ti.template()
):
    """
    @taichi.func

    Compute the structure tensor. 

    Args:
      Static:
        `k_ext`: ti.field(dtype=[float], shape=2*`radius_ext`+1) first order
          Gaussian derivative kernel.
        `radius_ext`: radius at which kernel `k_ext` is truncated, taking
          integer values greater than 0.
      Mutated:
        `d_d*`: ti.field(dtype=[float], shape=[Nx+2*`radius_ext`, Ny+2*`radius_ext`])
          Gaussian derivatives, which are updated in place.
        `Jρ_padded`: ti.field(dtype=[float], shape=[Nx+2*`radius_ext`, Ny+2*`radius_ext`])
          padded array to hold intermediate computations for the structure
          tensor.
        `Jρ**`: ti.field(dtype=[float], shape=[Nx, Ny]) **-component of the
          regularised structure tensor.
    """
    # Compute Jρ_11.
    for I in ti.grouped(Jρ_padded):
        Jρ_padded[I] = d_dx[I]**2
    convolve_with_kernel_x_dir(Jρ_padded, k_ext, radius_ext, Jρ11)
    convolve_with_kernel_y_dir(Jρ_padded, k_ext, radius_ext, Jρ11)
    # Compute Jρ_12.
    for I in ti.grouped(Jρ_padded):
        Jρ_padded[I] = d_dx[I] * d_dy[I]
    convolve_with_kernel_x_dir(Jρ_padded, k_ext, radius_ext, Jρ12)
    convolve_with_kernel_y_dir(Jρ_padded, k_ext, radius_ext, Jρ12)
    # Compute Jρ_22.
    for I in ti.grouped(Jρ_padded):
        Jρ_padded[I] = d_dy[I]**2
    convolve_with_kernel_x_dir(Jρ_padded, k_ext, radius_ext, Jρ22)
    convolve_with_kernel_x_dir(Jρ_padded, k_ext, radius_ext, Jρ22)


@ti.func
def S_ε_field(
    u: ti.template(),
    ε: ti.f32,
    S_ε_of_u: ti.template()
):
    """
    @taichi.func
    
    Compute Sε, the regularised signum as seen in "Regularised Diffusion-Shock 
    Inpainting" (2023) by K. Schaefer and J. Weickert.

    Args:
      Static:
        `u`: ti.field(dtype=ti.f32) to pass through regularised signum.
        `ε`: regularisation parameter, taking values greater than 0.
      Mutated:
        `S_ε_of_u`: ti.field(dtype=ti.f32) of S`ε`(`u`).
    """
    for I in ti.grouped(u):
        S_ε_of_u[I] = S_ε_scalar(u[I], ε)

@ti.func
def S_ε_scalar(
    x: ti.f32,
    ε: ti.f32
) -> ti.f32:
    """
    @taichi.func
    
    Compute Sε, the regularised signum as seen in "Regularised Diffusion-Shock 
    Inpainting" (2023) by K. Schaefer and J. Weickert.

    Args:
        `x`: scalar to pass through regularised signum, taking values greater 
          than 0.
        `ε`: regularisation parameter, taking values greater than 0.

    Returns:
        ti.f32 of S`ε`(`x`).
    """
    return (2 / ti.math.pi) * ti.math.atan(x, ε)

# Derivatives

@ti.func
def sobel_gradient(
    u: ti.template(),
    dxy: ti.f32,
    dx_u: ti.template(),
    dy_u: ti.template()
):
    """
    @taichi.func
    
    Compute approximations of the first order derivatives of `u` in the x and y 
    direction using Sobel operators, as described in Eq. (26) of "Regularised 
    Diffusion-Shock Inpainting" (2023) by K. Schaefer and J. Weickert.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `d*_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of d* `u`, which is
          updated in place.    
    """
    I_dx = ti.Vector([1, 0], dt=ti.i32)
    I_dy = ti.Vector([0, 1], dt=ti.i32)
    I_dplus = I_dx + I_dy  # Positive diagonal
    I_dminus = I_dx - I_dy # Negative diagonal
    for I in ti.grouped(u):
        I_dx_forward = sanitize_index(I + I_dx, u)
        I_dx_backward = sanitize_index(I - I_dx, u)
        I_dy_forward = sanitize_index(I + I_dy, u)
        I_dy_backward = sanitize_index(I - I_dy, u)
        I_dplus_forward = sanitize_index(I + I_dplus, u)
        I_dplus_backward = sanitize_index(I - I_dplus, u)
        I_dminus_forward = sanitize_index(I + I_dminus, u)
        I_dminus_backward = sanitize_index(I - I_dminus, u)
        # du/dx Stencil
        # -1 | 0 | 1
        # -2 | 0 | 2
        # -1 | 0 | 1
        dx_u[I] = (
            -1 * u[I_dminus_backward] +
            -2 * u[I_dx_backward] +
            -1 * u[I_dplus_backward] +
            1 * u[I_dminus_forward] +
            2 * u[I_dx_forward] +
            1 * u[I_dplus_forward]
        ) / (8 * dxy)
        # du/dy Stencil
        #  1 |  2 |  1
        #  0 |  0 |  0
        # -1 | -2 | -1
        dy_u[I] = (
            -1 * u[I_dplus_backward] +
            -2 * u[I_dy_backward] +
            -1 * u[I_dminus_forward] +
            1 * u[I_dminus_backward] +
            2 * u[I_dy_forward] +
            1 * u[I_dplus_forward]
        ) / (8 * dxy)