"""
    derivatives
    ===========

    Provides a variety of derivative operators on SE(2), namely:
      1. `laplacian`: 
      2. `morphological`: 
"""

import taichi as ti
from dsfilter.SE2.utils import scalar_trilinear_interpolate
from dsfilter.utils import (
    select_upwind_derivative_dilation,
    select_upwind_derivative_erosion
)
# Actual Derivatives

@ti.kernel
def laplacian(
    u: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    laplacian_u: ti.template()
):
    """
    @taichi.kernel

    Compute an approximation of the Laplace-Beltrami operator applied to `u`
    using central differences.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to
          differentiate.
        `G_inv`: ti.types.vector(n=3, dtype=[float]) constants of the inverse of
          the diagonal metric tensor with respect to left invariant basis.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `laplacian_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) laplacian of
          u, which is updated in place.

    References:
        [1]: K. Schaefer and J. Weickert.
          "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods
          in Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1137/15M1018460.
    """
    I_A3 = ti.Vector([0.0,  0.0, 1.0], dt=ti.f32)
    for I in ti.grouped(laplacian_u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A1 = ti.Vector([cos, sin, 0.0], dt=ti.f32)
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32)

        A11 = (scalar_trilinear_interpolate(u, I + I_A1) -
               2 * u[I] +
               scalar_trilinear_interpolate(u, I - I_A1)) / dxy**2
        A22 = (scalar_trilinear_interpolate(u, I + I_A2) -
               2 * u[I] +
               scalar_trilinear_interpolate(u, I - I_A2)) / dxy**2
        A33 = (scalar_trilinear_interpolate(u, I + I_A3) -
               2 * u[I] +
               scalar_trilinear_interpolate(u, I - I_A3)) / dθ**2
        # Δu = div(grad(u)) = A_i (g^ij A_j u) = g^ij A_i A_j u
        laplacian_u[I] = G_inv[0] * A11 + G_inv[1] * A22 + G_inv[2] * A33

@ti.kernel
def morphological(
    u: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    dilation_u: ti.template(),
    erosion_u: ti.template()
):
    """
    @taichi.kernel

    Compute upwind approximations of the morphological derivatives
    +/- ||grad `u`||.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx+2, Ny+2, Nθ]) which we want to 
          differentiate.
        `G_inv`: ti.types.vector(n=3, dtype=[float]) constants of the inverse of
          the diagonal metric tensor with respect to left invariant basis.
        `θs`: angle coordinate at each grid point.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `dilation_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) ||grad `u`||,
          which is updated in place.
        `erosion_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) -||grad `u`||,
          which is updated in place.
          
    References:
        [1]: K. Schaefer and J. Weickert.
          "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods in
          Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1137/15M1018460.
    """
    I_A3 = ti.Vector([0.0,  0.0, 1.0], dt=ti.f32)
    for I in ti.grouped(dilation_u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A1 = ti.Vector([cos, sin, 0.0], dt=ti.f32)
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32)

        A1_forward = (scalar_trilinear_interpolate(u, I + I_A1) - u[I]) / dxy
        A2_forward = (scalar_trilinear_interpolate(u, I + I_A2) - u[I]) / dxy
        A3_forward = (scalar_trilinear_interpolate(u, I + I_A3) - u[I]) / dθ
        A1_backward = (u[I] - scalar_trilinear_interpolate(u, I - I_A1)) / dxy
        A2_backward = (u[I] - scalar_trilinear_interpolate(u, I - I_A2)) / dxy
        A3_backward = (u[I] - scalar_trilinear_interpolate(u, I - I_A3)) / dθ

        # ||grad u|| = sqrt(G(grad u, grad u)) = sqrt(g^ij A_i u A_j u)
        # Dilation
        dilation_u[I] = ti.math.sqrt(
            G_inv[0] * select_upwind_derivative_dilation(A1_forward, A1_backward)**2 +
            G_inv[1] * select_upwind_derivative_dilation(A2_forward, A2_backward)**2 +
            G_inv[2] * select_upwind_derivative_dilation(A3_forward, A3_backward)**2
        )
        # Erosion
        erosion_u[I] = -ti.math.sqrt(
            G_inv[0] * select_upwind_derivative_erosion(A1_forward, A1_backward)**2 +
            G_inv[1] * select_upwind_derivative_erosion(A2_forward, A2_backward)**2 +
            G_inv[2] * select_upwind_derivative_erosion(A3_forward, A3_backward)**2
        )
        # erosion_u[I] = -ti.math.sqrt(
        #     G_inv[0] * select_upwind_derivative_dilation(-A1_forward, -A1_backward)**2 +
        #     G_inv[1] * select_upwind_derivative_dilation(-A2_forward, -A2_backward)**2 +
        #     G_inv[2] * select_upwind_derivative_dilation(-A3_forward, -A3_backward)**2
        # )

@ti.func
def gradient_perp(
    u: ti.template(),
    dxy: ti.f32,
    θs: ti.template(),
    gradient_perp_u: ti.template()
):
    """
    @taichi.func

    Compute an approximation of the perpendicular gradient of `u` using central
    differences.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to
          differentiate.
        `θs`: angle coordinate at each grid point.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `gradient_perp_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ])
          perpendicular gradient of u, which is updated in place.

    References:
        [1]: K. Schaefer and J. Weickert.
          "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods
          in Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1137/15M1018460.
    """
    for I in ti.grouped(gradient_perp_u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32)
        # grad_perp u = A_2 u A_2
        gradient_perp_u[I] = (
            scalar_trilinear_interpolate(u, I + I_A2) - scalar_trilinear_interpolate(u, I - I_A2)
        ) / (2 * dxy)

@ti.func
def laplace_perp(
    u: ti.template(),
    dxy: ti.f32,
    θs: ti.template(),
    laplace_perp_u: ti.template()
):
    """
    @taichi.func

    Compute an approximation of the perpendicular laplacian of `u` using central
    differences.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to
          differentiate.
        `θs`: angle coordinate at each grid point.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `laplace_perp_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ])
          perpendicular laplacian of u, which is updated in place.

    References:
        [1]: K. Schaefer and J. Weickert.
          "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods
          in Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1137/15M1018460.
    """
    for I in ti.grouped(laplace_perp_u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32)
        # Δ_perp u = A_2 A_2 u
        laplace_perp_u[I] = (
            scalar_trilinear_interpolate(u, I + I_A2) - 2 * u[I] + scalar_trilinear_interpolate(u, I - I_A2)
        ) / dxy**2
        # Δu = div(grad(u)) = A_i (g^ij A_j u) = g^ij A_i A_j u