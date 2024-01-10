# switches.py

import taichi as ti

# Diffusion-Shock

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
        `λ`: contrast parameters, taking values greater than 0.
      Mutated:
        ti.field(dtype=ti.f32) of g(`field_squared`).
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
        `λ`: contrast parameters, taking values greater than 0.

    Returns:
        ti.f32 of g(`s_squared`).
    """
    return 1 / ti.math.sqrt(1 + s_squared / λ**2)

# @ti.func
# def gaussian_dilation()

# Morphological

