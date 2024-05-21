"""
    frame
    =====

    Compute the gauge frames fitted to data on SE(2), using:
      1. `compute_gauge_frame`: 
"""

import taichi as ti
import numpy as np
from dsfilter.SE2.utils import scalar_trilinear_interpolate
from dsfilter.SE2.regularisers import (
    convolve_with_kernel_x_dir,
    convolve_with_kernel_y_dir,
    convolve_with_kernel_θ_dir,
    convolve_matrix_3_by_3_with_kernel_x_dir,
    convolve_matrix_3_by_3_with_kernel_y_dir,
    convolve_matrix_3_by_3_with_kernel_θ_dir
)


A = ti.Matrix(np.diag((-1, 2, 1)), dt=ti.f32)
print(A)

@ti.kernel
def test_eig(
    mat: ti.types.matrix(m=3, n=3, dtype=ti.f32)
) -> ti.types.matrix(m=3, n=3, dtype=ti.f32):
    eigenvalues, eigenvectors = ti.sym_eig(A)
    return eigenvectors

@ti.kernel
def test_eig_2(
    mat: ti.types.matrix(m=3, n=3, dtype=ti.f32)
) -> ti.types.vector(n=3, dtype=ti.f32):
    eigenvalues, eigenvectors = ti.sym_eig(A)
    return eigenvalues

test_eig_2(A)

@ti.func
def compute_gauge_frame(
    u: ti.template(),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    k_int_s: ti.template(),
    radius_int_s: ti.i32,
    k_int_o: ti.template(),
    radius_int_o: ti.i32,
    k_ext_s: ti.template(),
    radius_ext_s: ti.i32,
    k_ext_o: ti.template(),
    radius_ext_o: ti.i32,
    ξ: ti.f32,
    A1_u: ti.template(),
    A2_u: ti.template(),
    A3_u: ti.template(),
    H: ti.template(),
    A: ti.template(),
    V: ti.template(),
    B1: ti.template(),
    B2: ti.template(),
    B3: ti.template(),
    convolution_storage_1: ti.template(),
    convolution_storage_2: ti.template(),
    convolution_matrix_storage_1: ti.template(),
    convolution_matrix_storage_2: ti.template()
):
    """
    @taichi.kernel

    Compute gauge frame {B1, B2, B3}, with respect to the left invariant frame.
    TODO: add in reference to some paper; what one has the best explanation?

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
        `θs`: angle coordinate at each grid point.
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
        `laplacian_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) laplacian of
          u, which is updated in place.
    """
    # First regularise internally with Gaussian convolution.
    convolve_with_kernel_x_dir(u, k_int_s, radius_int_s, convolution_storage_1)
    convolve_with_kernel_y_dir(convolution_storage_1, k_int_s, radius_int_s, convolution_storage_2)
    convolve_with_kernel_θ_dir(convolution_storage_2, k_int_o, radius_int_o, u)
    # Then compute Hessian matrix.
    compute_Hessian_matrix(u, dxy, dθ, θs, A1_u, A2_u, A3_u, H)
    # Finally regularise componentwise externally with Gaussian convolutions.
    convolve_matrix_3_by_3_with_kernel_x_dir(H, k_ext_s, radius_ext_s, convolution_matrix_storage_1)
    convolve_matrix_3_by_3_with_kernel_y_dir(convolution_matrix_storage_1, k_ext_s, radius_ext_s, convolution_matrix_storage_2)
    convolve_matrix_3_by_3_with_kernel_θ_dir(convolution_matrix_storage_2, k_ext_o, radius_ext_o, H)
    # Make the problem dimensionless:
    # The spatial directions have dimensions [length], while the orientational
    # direction is dimensionless. To be able to compare them, we need to make
    # a choice of metric tensor field. We choose a spatially isotropic metric
    # tensor field Mξ with stiffness parameter ξ.
    Mξ = ti.Matrix(np.diag((1/ξ, 1/ξ, 1.)), dt=ti.f32)
    for I in ti.grouped(A):
        A[I] = H[I].transpose() * Mξ**2 * H[I]
    # Find the eigenvectors of the dimensionless problem.
    _, V = ti.sym_eig(A)
    # TODO: Consider other ways of getting B2 and B3:
    # Currently use only the eigenspace with smallest eigenvalue. However, since
    # the matrix is symmetric positive-definite, we know that it has orthogonal
    # eigenspaces. Hence, we could choose B2, B3 to simply be in the other
    # eigenspaces, instead of forcing B2 to be purely spatial.
    for I in ti.grouped(B1):
        # The main gauge vector is chosen to be in the eigenspace with the
        # smallest eigenvalue.
        c1 = V[I][0, 2]
        sign = ti.math.sign(c1)
        c1 *= sign
        c2 = V[I][1, 2] * sign
        c3 = V[I][2, 2] * sign
        # Spatial angle of B1, which may differ from θs[I], so that A1 and B1
        # do not point in the same direction spatially (deviation from
        # horizontality).
        χ = ti.math.atan2(c2, c1)
        cosχ = ti.math.cos(χ)
        sinχ = ti.math.sin(χ)
        # Orientational angle of B1, 
        ν = ti.math.atan2(c3, ti.math.sqrt(c1**2 + c2**2))
        cosν = ti.math.cos(ν)
        sinν = ti.math.sin(ν)

        B1[I][0] = c1
        B1[I][1] = c2
        B1[I][2] = c3
        B2[I][0] = -sinχ / ξ
        B2[I][1] = cosχ / ξ
        B2[I][2] = 0
        B3[I][0] = -cosχ * sinν / ξ
        B3[I][1] = -sinχ * sinν / ξ
        B3[I][2] = cosν



@ti.func
def compute_Hessian_matrix(
    u: ti.template(),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    A1_u: ti.template(),
    A2_u: ti.template(),
    A3_u: ti.template(),
    H: ti.template()
):
    """
    @taichi.func

    Compute an approximation of the Hessian matrix H^i_j `u` = A_j A_i `u`, the
    components of the Lie-Cartan 0 connection Hessian with respect to the left
    invariant frame.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `A*_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of derivatives, which
          are updated in place.
        `H`: ti.Matrix.field(m=3, n=3, dtype=[float], shape=[Nx, Ny, Nθ]) field
          of Lie-Cartan 0 connection Hessian matrices with respect to the left
          invariant frame, which is updated in place.
    """
    I_A3 = ti.Vector([0.0,  0.0, 1.0], dt=ti.f32)/2 # We do 2 first order central differences.
    # First order derivatives.
    for I in ti.grouped(u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A1 = ti.Vector([cos, sin, 0.0], dt=ti.f32)/2
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32)/2
        A1_u[I] = (scalar_trilinear_interpolate(u, I + I_A1) - scalar_trilinear_interpolate(u, I - I_A1)) / dxy
        A2_u[I] = (scalar_trilinear_interpolate(u, I + I_A2) - scalar_trilinear_interpolate(u, I - I_A2)) / dxy
        A3_u[I] = (scalar_trilinear_interpolate(u, I + I_A3) - scalar_trilinear_interpolate(u, I - I_A3)) / dθ
    # Second order mixed derivatives.
    for I in ti.grouped(H):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A1 = ti.Vector([cos, sin, 0.0], dt=ti.f32)/2
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32)/2
        H[I][0, 0] = (scalar_trilinear_interpolate(A1_u, I + I_A1) - scalar_trilinear_interpolate(A1_u, I - I_A1)) / dxy # A11
        H[I][0, 1] = (scalar_trilinear_interpolate(A1_u, I + I_A2) - scalar_trilinear_interpolate(A1_u, I - I_A2)) / dxy # A21
        H[I][0, 2] = (scalar_trilinear_interpolate(A1_u, I + I_A3) - scalar_trilinear_interpolate(A1_u, I - I_A3)) / dθ  # A31
        H[I][1, 0] = (scalar_trilinear_interpolate(A2_u, I + I_A1) - scalar_trilinear_interpolate(A2_u, I - I_A1)) / dxy # A12
        H[I][1, 1] = (scalar_trilinear_interpolate(A2_u, I + I_A2) - scalar_trilinear_interpolate(A2_u, I - I_A2)) / dxy # A22
        H[I][1, 2] = (scalar_trilinear_interpolate(A2_u, I + I_A3) - scalar_trilinear_interpolate(A2_u, I - I_A3)) / dθ  # A32
        H[I][2, 0] = (scalar_trilinear_interpolate(A3_u, I + I_A1) - scalar_trilinear_interpolate(A3_u, I - I_A1)) / dxy # A13
        H[I][2, 1] = (scalar_trilinear_interpolate(A3_u, I + I_A2) - scalar_trilinear_interpolate(A3_u, I - I_A2)) / dxy # A23
        H[I][2, 2] = (scalar_trilinear_interpolate(A3_u, I + I_A3) - scalar_trilinear_interpolate(A3_u, I - I_A3)) / dθ  # A33