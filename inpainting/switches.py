# switches.py

import taichi as ti
import inpainting

# Diffusion-Shock

## Switcher

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
        dx_u[I] = 1 / (8 * dxy) * (
            -1 * u[I_dminus_backward] +
            -2 * u[I_dx_backward] +
            -1 * u[I_dplus_backward] +
            1 * u[I_dminus_forward] +
            2 * u[I_dx_forward] +
            1 * u[I_dplus_forward]
        )
        # du/dy Stencil
        #  1 |  2 |  1
        #  0 |  0 |  0
        # -1 | -2 | -1
        dy_u[I] = 1 / (8 * dxy) * (
            -1 * u[I_dplus_backward] +
            -2 * u[I_dy_backward] +
            -1 * u[I_dminus_forward] +
            1 * u[I_dminus_backward] +
            2 * u[I_dy_forward] +
            1 * u[I_dplus_forward]
        )
            

    

# @ti.func
# def gaussian_dilation()

# Morphological
    

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
        `S_ε_of_u`: ti.field(dtype=ti.f32) of Sε(`u`).
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
        ti.f32 of Sε(`x`).
    """
    return (2 / ti.math.pi) * ti.math.atan(x, ε)