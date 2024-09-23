"""
    derivatives
    ===========

    Provides a variety of derivative operators on SE(2), namely:
      1. `laplacian`: 
      2. `morphological`: 
"""

import taichi as ti
from dsfilter.SE2.utils import scalar_trilinear_interpolate
from dsfilter.SE2.regularisers import (
    convolve_with_kernel_x_dir,
    convolve_with_kernel_y_dir,
    convolve_with_kernel_θ_dir
)
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
    ξ: ti.f32,
    B1: ti.template(),
    B2: ti.template(),
    B3: ti.template(),
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
    """
    h = ξ * dxy
    for I in ti.grouped(laplacian_u):
        # Split gauge vectors into spatial and orientational part, so that we 
        # can take steps that are roughly 1 pixel in size.
        I_B1_s = ti.Vector([B1[I][0] * ξ, B1[I][1] * ξ, 0.], dt=ti.f32)
        I_B1_o = ti.Vector([0., 0., B1[I][2]])
        I_B2_s = ti.Vector([B2[I][0] * ξ, B2[I][1] * ξ, 0.], dt=ti.f32)
        I_B2_o = ti.Vector([0., 0., B2[I][2]])
        I_B3_s = ti.Vector([B3[I][0] * ξ, B3[I][1] * ξ, 0.], dt=ti.f32)
        I_B3_o = ti.Vector([0., 0., B3[I][2]])

        B11 = (
            (scalar_trilinear_interpolate(u, I + I_B1_s) -
             2 * u[I] +
             scalar_trilinear_interpolate(u, I - I_B1_s)) / h**2 + 
            (scalar_trilinear_interpolate(u, I + I_B1_o) -
             2 * u[I] +
             scalar_trilinear_interpolate(u, I - I_B1_o)) / dθ**2
        )
        B22 = (
            (scalar_trilinear_interpolate(u, I + I_B2_s) -
             2 * u[I] +
             scalar_trilinear_interpolate(u, I - I_B2_s)) / h**2 + 
            (scalar_trilinear_interpolate(u, I + I_B2_o) -
             2 * u[I] +
             scalar_trilinear_interpolate(u, I - I_B2_o)) / dθ**2
        )
        B33 = (
            (scalar_trilinear_interpolate(u, I + I_B3_s) -
             2 * u[I] +
             scalar_trilinear_interpolate(u, I - I_B3_s)) / h**2 + 
            (scalar_trilinear_interpolate(u, I + I_B3_o) -
             2 * u[I] +
             scalar_trilinear_interpolate(u, I - I_B3_o)) / dθ**2
        )
        # Δu = div(grad(u)) = sqrt(det(g)) B_i (sqrt(det(g)) g^ij B_j u) = g^ij B_i B_j u = g^ii B_i B_i u
        laplacian_u[I] = G_inv[0] * B11 + G_inv[1] * B22 + G_inv[2] * B33

@ti.kernel
def morphological(
    u: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    ξ: ti.f32,
    B1: ti.template(),
    B2: ti.template(),
    B3: ti.template(),
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
    """
    h = ξ * dxy
    for I in ti.grouped(dilation_u):
        # Split gauge vectors into spatial and orientational part, so that we 
        # can take steps that are roughly 1 pixel in size.
        I_B1_s = ti.Vector([B1[I][0] * ξ, B1[I][1] * ξ, 0.], dt=ti.f32)
        I_B1_o = ti.Vector([0., 0., B1[I][2]])
        I_B2_s = ti.Vector([B2[I][0] * ξ, B2[I][1] * ξ, 0.], dt=ti.f32)
        I_B2_o = ti.Vector([0., 0., B2[I][2]])
        I_B3_s = ti.Vector([B3[I][0] * ξ, B3[I][1] * ξ, 0.], dt=ti.f32)
        I_B3_o = ti.Vector([0., 0., B3[I][2]])

        B1_forward = (
            (scalar_trilinear_interpolate(u, I + I_B1_s) - u[I]) / h +
            (scalar_trilinear_interpolate(u, I + I_B1_o) - u[I]) / dθ
        )
        B2_forward = (
            (scalar_trilinear_interpolate(u, I + I_B2_s) - u[I]) / h +
            (scalar_trilinear_interpolate(u, I + I_B2_o) - u[I]) / dθ
        )
        B3_forward = (
            (scalar_trilinear_interpolate(u, I + I_B3_s) - u[I]) / h +
            (scalar_trilinear_interpolate(u, I + I_B3_o) - u[I]) / dθ
        )
        B1_backward = (
            (u[I] - scalar_trilinear_interpolate(u, I - I_B1_s)) / h +
            (u[I] - scalar_trilinear_interpolate(u, I - I_B1_o)) / dθ
        )
        B2_backward = (
            (u[I] - scalar_trilinear_interpolate(u, I - I_B2_s)) / h +
            (u[I] - scalar_trilinear_interpolate(u, I - I_B2_o)) / dθ
        )
        B3_backward = (
            (u[I] - scalar_trilinear_interpolate(u, I - I_B3_s)) / h +
            (u[I] - scalar_trilinear_interpolate(u, I - I_B3_o)) / dθ
        )

        # ||grad u|| = sqrt(G(grad u, grad u)) = sqrt(g^ij B_i u B_j u) = sqrt(g^ii (B_i u)^2)
        # Dilation
        dilation_u[I] = ti.math.sqrt(
            G_inv[0] * select_upwind_derivative_dilation(B1_forward, B1_backward)**2 +
            G_inv[1] * select_upwind_derivative_dilation(B2_forward, B2_backward)**2 +
            G_inv[2] * select_upwind_derivative_dilation(B3_forward, B3_backward)**2
        )
        # Erosion
        erosion_u[I] = -ti.math.sqrt(
            G_inv[0] * select_upwind_derivative_erosion(B1_forward, B1_backward)**2 +
            G_inv[1] * select_upwind_derivative_erosion(B2_forward, B2_backward)**2 +
            G_inv[2] * select_upwind_derivative_erosion(B3_forward, B3_backward)**2
        )

@ti.func
def gradient_perp(
    u: ti.template(),
    dxy: ti.f32,
    dθ: ti.f32,
    ξ: ti.f32,
    B2: ti.template(),
    B3: ti.template(),
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
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
        `θs`: angle coordinate at each grid point.
        `ξ`: stiffness parameter defining the cost of moving one unit in the
          orientatonal direction relative to moving one unit in a spatial
          direction, taking values greater than 0.
      Mutated:
        `gradient_perp_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ])
          perpendicular gradient of u, which is updated in place.
    """
    h = ξ * dxy
    for I in ti.grouped(gradient_perp_u):
        # Split gauge vectors into spatial and orientational part, so that we 
        # can take steps that are roughly 1 pixel in size.
        I_B2_s = ti.Vector([B2[I][0] * ξ, B2[I][1] * ξ, 0.], dt=ti.f32) / 2
        I_B2_o = ti.Vector([0., 0., B2[I][2]]) / 2
        I_B3_s = ti.Vector([B3[I][0] * ξ, B3[I][1] * ξ, 0.], dt=ti.f32) / 2
        I_B3_o = ti.Vector([0., 0., B3[I][2]]) / 2
        # ||grad_perp u|| = sqrt(G(grad_perp u, grad_perp u)) = sqrt(g^ij B_i u B_j u) = sqrt(g^22 (B_2 u)^2 + g^33 (B_3 u)^2)
        gradient_perp_u[I] = ti.math.sqrt(((
                scalar_trilinear_interpolate(u, I + I_B2_s) - scalar_trilinear_interpolate(u, I - I_B2_s)
            ) / h + (
                scalar_trilinear_interpolate(u, I + I_B2_o) - scalar_trilinear_interpolate(u, I - I_B2_o)
            ) / dθ)**2 + ((
                scalar_trilinear_interpolate(u, I + I_B3_s) - scalar_trilinear_interpolate(u, I - I_B3_s)
            ) / h + (
                scalar_trilinear_interpolate(u, I + I_B3_o) - scalar_trilinear_interpolate(u, I - I_B3_o)
            ) / dθ)**2
        )

@ti.func
def laplace_perp(
    u: ti.template(),
    dxy: ti.f32,
    dθ: ti.f32,
    ξ: ti.f32,
    B2: ti.template(),
    B3: ti.template(),
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
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
        `θs`: angle coordinate at each grid point.
        `ξ`: stiffness parameter defining the cost of moving one unit in the
          orientatonal direction relative to moving one unit in a spatial
          direction, taking values greater than 0.
      Mutated:
        `laplace_perp_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ])
          perpendicular laplacian of u, which is updated in place.
    """
    h = ξ * dxy
    for I in ti.grouped(laplace_perp_u):
        I_B2_s = ti.Vector([B2[I][0] * ξ, B2[I][1] * ξ, 0.], dt=ti.f32)
        I_B2_o = ti.Vector([0., 0., B2[I][2]])
        I_B3_s = ti.Vector([B3[I][0] * ξ, B3[I][1] * ξ, 0.], dt=ti.f32)
        I_B3_o = ti.Vector([0., 0., B3[I][2]])
        # Δ_perp u = div_perp(grad_perp(u)) = sqrt(det(g)) B_i (sqrt(det(g)) g^ij B_j u) = g^ij B_i B_j u = B_2 B_2 u + B_3 B_3 u
        laplace_perp_u[I] = ((
                scalar_trilinear_interpolate(u, I + I_B2_s) - 2 * u[I] + scalar_trilinear_interpolate(u, I - I_B2_s)
            ) / h**2 + (
                scalar_trilinear_interpolate(u, I + I_B2_o) - 2 * u[I] + scalar_trilinear_interpolate(u, I - I_B2_o)
            ) / dθ**2 + (
                scalar_trilinear_interpolate(u, I + I_B3_s) - 2 * u[I] + scalar_trilinear_interpolate(u, I - I_B3_s)
            ) / h**2 + (
                scalar_trilinear_interpolate(u, I + I_B3_o) - 2 * u[I] + scalar_trilinear_interpolate(u, I - I_B3_o)
            ) / dθ**2
        )

@ti.kernel
def TV(
    u: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    ξ: ti.f32,
    B1: ti.template(),
    B2: ti.template(),
    B3: ti.template(),
    k_s: ti.template(),
    radius_s: ti.template(),
    k_o: ti.template(),
    radius_o: ti.template(),
    B1_u: ti.template(),
    B2_u: ti.template(),
    B3_u: ti.template(),
    grad_norm_u: ti.template(),
    normalised_grad_1: ti.template(),
    normalised_grad_2: ti.template(),
    normalised_grad_3: ti.template(),
    TV_u: ti.template(),
    storage: ti.template()
):
    """
    @taichi.kernel

    Compute an approximation of the Total Variation (TV) operator applied to `u`
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
        `TV_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) laplacian of
          u, which is updated in place.
    """
    h = ξ * dxy
    for I in ti.grouped(B1_u):
        # Split gauge vectors into spatial and orientational part, so that we 
        # can take steps that are roughly 1 pixel in size.
        I_B1_s = ti.Vector([B1[I][0] * ξ, B1[I][1] * ξ, 0.], dt=ti.f32) / 2
        I_B1_o = ti.Vector([0., 0., B1[I][2]]) / 2
        I_B2_s = ti.Vector([B2[I][0] * ξ, B2[I][1] * ξ, 0.], dt=ti.f32) / 2
        I_B2_o = ti.Vector([0., 0., B2[I][2]]) / 2
        I_B3_s = ti.Vector([B3[I][0] * ξ, B3[I][1] * ξ, 0.], dt=ti.f32) / 2
        I_B3_o = ti.Vector([0., 0., B3[I][2]]) / 2

        B1_u[I] = c1 = (
            (scalar_trilinear_interpolate(u, I + I_B1_s) - scalar_trilinear_interpolate(u, I - I_B1_s)) / h +
            (scalar_trilinear_interpolate(u, I + I_B1_o) - scalar_trilinear_interpolate(u, I - I_B1_o)) / dθ
        )
        B2_u[I] = c2 = (
            (scalar_trilinear_interpolate(u, I + I_B2_s) - scalar_trilinear_interpolate(u, I - I_B2_s)) / h +
            (scalar_trilinear_interpolate(u, I + I_B2_o) - scalar_trilinear_interpolate(u, I - I_B2_o)) / dθ
        )
        B3_u[I] = c3 = (
            (scalar_trilinear_interpolate(u, I + I_B3_s) - scalar_trilinear_interpolate(u, I - I_B3_s)) / h +
            (scalar_trilinear_interpolate(u, I + I_B3_o) - scalar_trilinear_interpolate(u, I - I_B3_o)) / dθ
        )
        grad_norm_u[I] = ti.math.sqrt(G_inv[0] * c1**2 + G_inv[1] * c2**2 + G_inv[2] * c3**2)
    convolve_with_kernel_x_dir(grad_norm_u, k_s, radius_s, TV_u)
    convolve_with_kernel_y_dir(TV_u, k_s, radius_s, storage)
    convolve_with_kernel_θ_dir(storage, k_o, radius_o, grad_norm_u)

    for I in ti.grouped(normalised_grad_1):
        normalised_grad_1[I] = G_inv[0] * B1_u[I] / grad_norm_u[I]
        normalised_grad_2[I] = G_inv[1] * B2_u[I] / grad_norm_u[I]
        normalised_grad_3[I] = G_inv[2] * B3_u[I] / grad_norm_u[I]

    for I in ti.grouped(TV_u):
        # Split gauge vectors into spatial and orientational part, so that we 
        # can take steps that are roughly 1 pixel in size.
        I_B1_s = ti.Vector([B1[I][0] * ξ, B1[I][1] * ξ, 0.], dt=ti.f32) / 2
        I_B1_o = ti.Vector([0., 0., B1[I][2]]) / 2
        I_B2_s = ti.Vector([B2[I][0] * ξ, B2[I][1] * ξ, 0.], dt=ti.f32) / 2
        I_B2_o = ti.Vector([0., 0., B2[I][2]]) / 2
        I_B3_s = ti.Vector([B3[I][0] * ξ, B3[I][1] * ξ, 0.], dt=ti.f32) / 2
        I_B3_o = ti.Vector([0., 0., B3[I][2]]) / 2
        divnormgrad1 = (
            (scalar_trilinear_interpolate(normalised_grad_1, I + I_B1_s) -
             scalar_trilinear_interpolate(normalised_grad_1, I - I_B1_s)) / h +
            (scalar_trilinear_interpolate(normalised_grad_1, I + I_B1_o) -
             scalar_trilinear_interpolate(normalised_grad_1, I - I_B1_o)) / dθ
        )
        divnormgrad2 = (
            (scalar_trilinear_interpolate(normalised_grad_2, I + I_B2_s) -
             scalar_trilinear_interpolate(normalised_grad_2, I - I_B2_s)) / h +
            (scalar_trilinear_interpolate(normalised_grad_2, I + I_B2_o) -
             scalar_trilinear_interpolate(normalised_grad_2, I - I_B2_o)) / dθ
        )
        divnormgrad3 = (
            (scalar_trilinear_interpolate(normalised_grad_3, I + I_B3_s) -
             scalar_trilinear_interpolate(normalised_grad_3, I - I_B3_s)) / h +
            (scalar_trilinear_interpolate(normalised_grad_3, I + I_B3_o) -
             scalar_trilinear_interpolate(normalised_grad_3, I - I_B3_o)) / dθ
        )
        TV_u[I] = divnormgrad1 + divnormgrad2 + divnormgrad3