# switches.py

import taichi as ti
import inpainting

# Diffusion-Shock

## Switcher

@ti.func
def DS_switch(
    u_padded: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    λ: ti.f32,
    d_dx: ti.template(),
    d_dy: ti.template(),
    switch: ti.template()
):
    """
    @taichi.func

    Determine to what degree we should perform diffusion or shock, as in
    "Diffusion-Shock Inpainting" (2023) by K. Schaefer and J. Weickert.

    Args:
      Static:
        `u_padded`: ti.field(dtype=ti.f32, shape=shape_padded) of array to be
          convolved, with shape_padded[i] = shape[i] + 2 * `radius`.
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) of first order Gaussian
          derivative kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
        `λ`: contrast parameter, taking values greater than 0.
      Mutated:
        `d_d*`: ti.field(dtype=ti.f32, shape=shape) of Gaussian derivatives,
          which are updated in place.
        `switch`: ti.field(dtype=ti.f32, shape=shape) of values that determine
          the degree of diffusion or shock, taking values between 0 and 1.
    """
    inpainting.derivativesR2.convolve_with_kernel_x_dir(u_padded, k, radius, d_dx)
    inpainting.derivativesR2.convolve_with_kernel_y_dir(u_padded, k, radius, d_dy)
    for I in ti.grouped(switch):
        switch[I] = g_scalar(d_dx[I]**2 + d_dy[I]**2, λ)

@ti.func
def g_field(
    field_squared: ti.template(), 
    λ: ti.f32,
    g_of_field_squared: ti.template()
):
    """
    @taichi.func
    
    Compute g, the function that switches between diffusion and shock in
    "Diffusion-Shock Inpainting" (2023) by K. Schaefer and J. Weickert.

    Args:
      Static:
        `field_squared`: ti.field(dtype=ti.f32) square of some scalar field; in 
          DS filtering the square of |grad Gν * u|, where Gν is the Gaussian 
          with standard deviation ν.
        `λ`: contrast parameter, taking values greater than 0.
      Mutated:
        `g_of_field_squared`: ti.field(dtype=ti.f32) of g(`field_squared`).
    """
    for I in ti.grouped(field_squared):
        g_of_field_squared[I] = g_scalar(field_squared[I], λ) 

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


## Derivatives

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
        `s_squared`: square of some scalar, taking values greater than 0.
        `λ`: contrast parameters, taking values greater than 0.
      Mutated:
      
    """
    I_dx = ti.Vector([1, 0], dt=ti.i32)
    I_dy = ti.Vector([0, 1], dt=ti.i32)
    I_dplus = I_dx + I_dy  # Positive diagonal
    I_dminus = I_dx - I_dy # Negative diagonal
    for I in ti.grouped(u):
        I_dx_forward = inpainting.sanitize_index(I + I_dx, u)
        I_dx_backward = inpainting.sanitize_index(I - I_dx, u)
        I_dy_forward = inpainting.sanitize_index(I + I_dy, u)
        I_dy_backward = inpainting.sanitize_index(I - I_dy, u)
        I_dplus_forward = inpainting.sanitize_index(I + I_dplus, u)
        I_dplus_backward = inpainting.sanitize_index(I - I_dplus, u)
        I_dminus_forward = inpainting.sanitize_index(I + I_dminus, u)
        I_dminus_backward = inpainting.sanitize_index(I - I_dminus, u)
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

# Morphological

## Switcher

@ti.func
def morphological_switch(
    u_padded: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    dxy: ti.f32,
    d_dx: ti.template(),
    d_dy: ti.template(),
    c: ti.template(),
    s: ti.template(),
    u: ti.template(),
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
        `u_padded`: ti.field(dtype=ti.f32, shape=shape_padded) of padded current
          state, with shape_padded[i] = shape[i] + 2 * `radius`.
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) of first order Gaussian
          derivative kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `u`: ti.field(dtype=ti.f32, shape=shape) of current state.
      Mutated:
        `d_d*`: ti.field(dtype=ti.f32) of first order Gaussian derivatives,
          which are updated in place.
        `d_d**`: ti.field(dtype=ti.f32) of second order Gaussian derivatives,
          which are updated in place.
        `switch`: ti.field(dtype=ti.f32, shape=shape) of values that determine
          the degree of dilation or erosion, taking values between -1 and 1.
    """
    find_dominant_eigenvector(u_padded, k, radius, d_dx, d_dy, c, s)
    inpainting.derivativesR2.central_derivatives_second_order(u, dxy, d_dxx, d_dxy, d_dyy)
    for I in ti.grouped(switch):
        switch[I] = ti.math.sign(c[I]**2 * d_dxx[I] + 2 * c[I] * s[I] * d_dxy[I] + s[I]**2 * d_dyy[I])


# !!!NO EXTERNAL REGULARISATION!!!
# Using that grad u is the dominant eigenvector of grad u grad u^T. We would 
# like to work with a regularised version of the structure tensor, namely
# Jρ := Gρ * (grad uσ grad uσ^T). However, then grad uσ is no longer the 
# dominant eigenvector, so we have to do actual work. We could use the matrix
# power method: take some random u0, and compute Jρ^n u0 for some large n. As 
# long as u0 is not fully in the span of the small eigenvector, the resulting
# vector will almost be in the span of the dominant eigenvector.

@ti.func
def find_dominant_eigenvector(
    u_padded: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    d_dx: ti.template(),
    d_dy: ti.template(),
    c: ti.template(),
    s: ti.template()
):
    """
    Compute the dominant eigenvector of the structure tensor, as in
    "Diffusion-Shock Inpainting" (2023) by K. Schaefer and J. Weickert.

    Args:
      Static:
        `u_padded`: ti.field(dtype=ti.f32, shape=shape_padded) of array of which
          the dominant eigenvectors are to be found, with
          shape_padded[i] = shape[i] + 2 * `radius`.
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) of first order Gaussian
          derivative kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `d_d*`: ti.field(dtype=ti.f32) of Gaussian derivatives, which are
          updated in place.
        `c`: ti.field(dtype=ti.f32, shape=shape) of first components of the
          normalised dominant eigenvectors, as in Eq. (15).
        `s`: ti.field(dtype=ti.f32, shape=shape) of second components of the
          normalised dominant eigenvectors, as in Eq. (15).
    """
    inpainting.derivativesR2.convolve_with_kernel_x_dir(u_padded, k, radius, d_dx)
    inpainting.derivativesR2.convolve_with_kernel_y_dir(u_padded, k, radius, d_dy)
    for I in ti.grouped(c):
        norm = ti.math.sqrt(d_dx[I]**2 + d_dy[I]**2)
        c[I] = d_dx[I] / norm
        s[I] = d_dy[I] / norm


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