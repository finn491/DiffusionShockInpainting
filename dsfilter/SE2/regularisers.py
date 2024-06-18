"""
    regularisers
    ============

    Provides tools to regularise scalar fields on SE(2), namely:
      1. `convolve_with_kernel_x_dir`: convolve a field with a 1D kernel along
      the x-direction.
      2. `convolve_with_kernel_y_dir`: convolve a field with a 1D kernel along
      the y-direction.
      3. `convolve_with_kernel_θ_dir`: convolve a field with a 1D kernel along
      the θ-direction.
      4. `gaussian_derivative_kernel`: computes 1D Gaussian derivative kernels
      of order 0 and 1, using an algorithm that improves the accuracy of higher
      order derivative kernels with small widths, based on the DIPlib[1]
      algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.
    We use that the spatially isotropic diffusion equation on SE(2) can be
    solved by convolving in the x-, y-, and θ-direction with some 1D kernel. For
    the x- and y-directions, this kernel is a Gaussian; for the θ-direction the
    kernel looks like a Gaussian if the amount of diffusion is sufficiently
    small.

    TODO: maybe add in correct kernel for θ-direction?

    References:
      [1]: C. Luengo, W. Caarls, R. Ligteringen, E. Schuitema, Y. Guo,
      E. Wernersson, F. Malmberg, S. Lokhorst, M. Wolff, G. van Kempen,
      M. van Ginkel, L. van Vliet, B. Rieger, B. Verwer, H. Netten,
      J. W. Brandenburg, J. Dijk, N. van den Brink, F. Faas, K. van Wijk,
      and T. Pham. "DIPlib 3". GitHub: https://github.com/DIPlib/diplib.
"""

import taichi as ti
from dsfilter.SE2.utils import (
    sanitize_reflected_index,
    scalar_trilinear_interpolate
)

# We cannot nest parallelised loops in if-else statements in TaiChi kernels.

@ti.func
def regularise_anisotropic(
    u: ti.template(),
    θs: ti.template(),
    dxy: ti.f32,
    dθ: ti.f32,
    σ1: ti.f32,
    σ2: ti.f32,
    σ3: ti.f32,
    u_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve `u` with the 1D kernel `k` in the x-direction.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) array to be convolved.
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) convolution
        of `u` with `k`.
    """
    radius = 2
    Ks = radius * ti.math.ceil(ti.math.max(σ1, σ2) / dxy, ti.i32)
    Ko = radius * ti.math.ceil(σ3 / dθ, ti.i32)
    norm = (
        ti.math.sqrt((2 * ti.math.pi)**3) * σ1 * σ2 * σ3 / # Normal distribution.
        (dxy * dxy * dθ) # Volume of a single voxel.
    )
    for I in ti.grouped(u):
        # Local orientation, along which one axis of the ellipsoid lies.
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        s = 0.
        Δx = -Ks * dxy
        for ix in range(2*Ks+1):
            Δy = -Ks * dxy
            for iy in range(2*Ks+1):
                Δθ = -Ko * dθ
                for iθ in range(2*Ko+1):
                    I_step = ti.Vector([Ks - ix, Ks - iy, Ko - iθ], ti.i32)
                    # Project onto axes of ellipsoid.
                    Δ1 = cos * Δx + sin * Δy
                    Δ2 = -sin * Δx + cos * Δy
                    Δ3 = Δθ
                    # Scaled distance to centre of kernel.
                    ρsq = (Δ1 / σ1)**2 + (Δ2 / σ2)**2 + (Δ3 / σ3)**2
                    diff = ti.math.exp(-ρsq/2) / norm
                    s += u[sanitize_reflected_index(I - I_step, u)] * diff
                    Δθ += dθ
                Δy += dxy 
            Δx += dxy
        u_convolved[I] = s

@ti.func
def convolve_with_kernel_x_dir(
    u: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    u_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve `u` with the 1D kernel `k` in the x-direction.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) array to be convolved.
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) convolution
        of `u` with `k`.
    """
    for x, y, θ in u_convolved:
        s = 0.
        for i in range(2*radius+1):
            index = sanitize_reflected_index(ti.Vector([x - radius + i, y, θ], dt=ti.i32), u)
            s += u[index] * k[2*radius-i]
        u_convolved[x, y, θ] = s

@ti.func
def convolve_with_kernel_y_dir(
    u: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    u_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve `u` with the 1D kernel `k` in the y-direction.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) array to be convolved.
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) convolution
          of `u` with `k`.
    """
    for x, y, θ in u_convolved:
        s = 0.
        for i in range(2*radius+1):
            index = sanitize_reflected_index(ti.Vector([x, y - radius + i, θ], dt=ti.i32), u)
            s+= u[index] * k[2*radius-i]
        u_convolved[x, y, θ] = s

@ti.func
def convolve_with_kernel_θ_dir(
    u: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    u_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve `u` with the 1D kernel `k` in the θ-direction.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) array to be convolved.
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) convolution
          of `u` with `k`.
    """
    for x, y, θ in u_convolved:
        s = 0.
        for i in range(2*radius+1):
            # This may in fact give the correct convolution...
            index = sanitize_reflected_index(ti.Vector([x, y, θ - radius + i], dt=ti.i32), u)
            s+= u[index] * k[2*radius-i]
        u_convolved[x, y, θ] = s

@ti.func
def convolve_matrix_3_by_3_with_kernel_x_dir(
    M: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    M_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve matrix field `M` with the 1D kernel `k` in the x-direction.

    Args:
      Static:
        `M`: ti.Matrix.field(m=3, n=3, dtype=[float], shape=[Nx, Ny, Nθ]) matrix
          field to be convolved.
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `M_convolved`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) convolution
          of `u` with `k`.
    """
    for x, y, θ in M_convolved:
        s00 = 0.
        s01 = 0.
        s02 = 0.
        s10 = 0.
        s11 = 0.
        s12 = 0.
        s20 = 0.
        s21 = 0.
        s22 = 0.
        for i in range(2*radius+1):
            index = sanitize_reflected_index(ti.Vector([x - radius + i, y, θ], dt=ti.i32), M)
            k_val = k[2*radius-i]
            s00 += M[index][0, 0] * k_val
            s01 += M[index][0, 1] * k_val
            s02 += M[index][0, 2] * k_val
            s10 += M[index][1, 0] * k_val
            s11 += M[index][1, 1] * k_val
            s12 += M[index][1, 2] * k_val
            s20 += M[index][2, 0] * k_val
            s21 += M[index][2, 1] * k_val
            s22 += M[index][2, 2] * k_val
        M_convolved[x, y, θ][0, 0] = s00
        M_convolved[x, y, θ][0, 1] = s01
        M_convolved[x, y, θ][0, 2] = s02
        M_convolved[x, y, θ][1, 0] = s10
        M_convolved[x, y, θ][1, 1] = s11
        M_convolved[x, y, θ][1, 2] = s12
        M_convolved[x, y, θ][2, 0] = s20
        M_convolved[x, y, θ][2, 1] = s21
        M_convolved[x, y, θ][2, 2] = s22

@ti.func
def convolve_matrix_3_by_3_with_kernel_y_dir(
    M: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    M_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve matrix field `M` with the 1D kernel `k` in the y-direction.

    Args:
      Static:
        `M`: ti.Matrix.field(m=3, n=3, dtype=[float], shape=[Nx, Ny, Nθ]) matrix
          field to be convolved.
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `M_convolved`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) convolution
          of `u` with `k`.
    """
    for x, y, θ in M_convolved:
        s00 = 0.
        s01 = 0.
        s02 = 0.
        s10 = 0.
        s11 = 0.
        s12 = 0.
        s20 = 0.
        s21 = 0.
        s22 = 0.
        for i in range(2*radius+1):
            index = sanitize_reflected_index(ti.Vector([x, y - radius + i, θ], dt=ti.i32), M)
            k_val = k[2*radius-i]
            s00 += M[index][0, 0] * k_val
            s01 += M[index][0, 1] * k_val
            s02 += M[index][0, 2] * k_val
            s10 += M[index][1, 0] * k_val
            s11 += M[index][1, 1] * k_val
            s12 += M[index][1, 2] * k_val
            s20 += M[index][2, 0] * k_val
            s21 += M[index][2, 1] * k_val
            s22 += M[index][2, 2] * k_val
        M_convolved[x, y, θ][0, 0] = s00
        M_convolved[x, y, θ][0, 1] = s01
        M_convolved[x, y, θ][0, 2] = s02
        M_convolved[x, y, θ][1, 0] = s10
        M_convolved[x, y, θ][1, 1] = s11
        M_convolved[x, y, θ][1, 2] = s12
        M_convolved[x, y, θ][2, 0] = s20
        M_convolved[x, y, θ][2, 1] = s21
        M_convolved[x, y, θ][2, 2] = s22

@ti.func
def convolve_matrix_3_by_3_with_kernel_θ_dir(
    M: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    M_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve matrix field `M` with the 1D kernel `k` in the θ-direction.

    Args:
      Static:
        `M`: ti.Matrix.field(m=3, n=3, dtype=[float], shape=[Nx, Ny, Nθ]) matrix
          field to be convolved.
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `M_convolved`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) convolution
          of `u` with `k`.
    """
    for x, y, θ in M_convolved:
        s00 = 0.
        s01 = 0.
        s02 = 0.
        s10 = 0.
        s11 = 0.
        s12 = 0.
        s20 = 0.
        s21 = 0.
        s22 = 0.
        for i in range(2*radius+1):
            index = sanitize_reflected_index(ti.Vector([x, y, θ - radius + i], dt=ti.i32), M)
            k_val = k[2*radius-i]
            s00 += M[index][0, 0] * k_val
            s01 += M[index][0, 1] * k_val
            s02 += M[index][0, 2] * k_val
            s10 += M[index][1, 0] * k_val
            s11 += M[index][1, 1] * k_val
            s12 += M[index][1, 2] * k_val
            s20 += M[index][2, 0] * k_val
            s21 += M[index][2, 1] * k_val
            s22 += M[index][2, 2] * k_val
        M_convolved[x, y, θ][0, 0] = s00
        M_convolved[x, y, θ][0, 1] = s01
        M_convolved[x, y, θ][0, 2] = s02
        M_convolved[x, y, θ][1, 0] = s10
        M_convolved[x, y, θ][1, 1] = s11
        M_convolved[x, y, θ][1, 2] = s12
        M_convolved[x, y, θ][2, 0] = s20
        M_convolved[x, y, θ][2, 1] = s21
        M_convolved[x, y, θ][2, 2] = s22

def gaussian_derivative_kernel(σ, order, truncate=5., dxy=1.):
    """Compute kernel for 1D Gaussian derivative of order `order` at scale `σ`.

    Based on the DIPlib[1] algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.

    Args:
        `σ`: scale of Gaussian, taking values greater than 0.
        `order`: order of the derivative, taking values 0 or 1.
        `truncate`: number of scales `σ` at which kernel is truncated, taking 
          values greater than 0.
        `dxy`: step size in x and y direction, taking values greater than 0.

    Returns:
        Tuple ti.field(dtype=[float], shape=2*radius+1) of the Gaussian kernel
          and the radius of the kernel.

    References:
        [1]: C. Luengo, W. Caarls, R. Ligteringen, E. Schuitema, Y. Guo,
          E. Wernersson, F. Malmberg, S. Lokhorst, M. Wolff, G. van Kempen,
          M. van Ginkel, L. van Vliet, B. Rieger, B. Verwer, H. Netten,
          J. W. Brandenburg, J. Dijk, N. van den Brink, F. Faas, K. van Wijk,
          and T. Pham. "DIPlib 3". GitHub: https://github.com/DIPlib/diplib.
    """
    radius = int(σ * truncate + 0.5)
    k = ti.field(dtype=ti.f32, shape=2*radius+1)
    match order:
        case 0:
            gaussian_derivative_kernel_order_0(σ, radius, dxy, k)
        case 1:
            gaussian_derivative_kernel_order_1(σ, radius, dxy, k)
        case _:
            raise(NotImplementedError(f"Order {order} has not been implemented yet; choose order 0 or 1."))
    return k, radius

@ti.kernel
def gaussian_derivative_kernel_order_0(
    σ: ti.f32,
    radius: ti.i32,
    dxy: ti.f32,
    k: ti.template()
):
    """
    @taichi.kernel
    
    Compute 1D Gaussian kernel at scale `σ`.

    Based on the DIPlib[1] algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.

    Args:
      Static:
        `σ`: scale of Gaussian, taking values greater than 0.
        `radius`: radius at which kernel is truncated, taking integer values
          greater than 0.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel, which is
          updated in place.

    References:
        [1]: C. Luengo, W. Caarls, R. Ligteringen, E. Schuitema, Y. Guo,
          E. Wernersson, F. Malmberg, S. Lokhorst, M. Wolff, G. van Kempen,
          M. van Ginkel, L. van Vliet, B. Rieger, B. Verwer, H. Netten,
          J. W. Brandenburg, J. Dijk, N. van den Brink, F. Faas, K. van Wijk,
          and T. Pham. "DIPlib 3". GitHub: https://github.com/DIPlib/diplib.
    """
    for i in range(2*radius+1):
        x = -radius + i
        val = ti.math.exp(-x**2 / (2 * σ**2))
        k[i] = val
    normalise_field(k, 1) # /dxy

@ti.kernel
def gaussian_derivative_kernel_order_1(
    σ: ti.f32,
    radius: ti.i32,
    dxy: ti.f32,
    k: ti.template()
):
    """
    @taichi.kernel
    
    Compute kernel for 1D Gaussian derivative of order 1 at scale `σ`.

    Based on the DIPlib[1] algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.

    Args:
      Static:
        `σ`: scale of Gaussian, taking values greater than 0.
        `radius`: radius at which kernel is truncated, taking integer values
          greater than 0.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel, which is
          updated in place.

    References:
        [1]: C. Luengo, W. Caarls, R. Ligteringen, E. Schuitema, Y. Guo,
          E. Wernersson, F. Malmberg, S. Lokhorst, M. Wolff, G. van Kempen,
          M. van Ginkel, L. van Vliet, B. Rieger, B. Verwer, H. Netten,
          J. W. Brandenburg, J. Dijk, N. van den Brink, F. Faas, K. van Wijk,
          and T. Pham. "DIPlib 3". GitHub: https://github.com/DIPlib/diplib.
    """
    moment = 0.
    for i in range(2*radius+1):
        x = -radius + i
        val = x * ti.math.exp(-x**2 / (2 * σ**2))
        moment += x * val
        k[i] = val
    divide_field(k, -moment * dxy)



# Helper Functions

@ti.func
def normalise_field(
    field: ti.template(),
    norm: ti.f32
):
    """
    @ti.func

    Normalise `field` to sum to `norm`.

    Args:
      Static:
        `norm`: desired norm for `field`, taking values greater than 0.
      Mutated:
        `field`: ti.field that is to be normalised, which is updated in place.    
    """
    current_norm = 0.
    for I in ti.grouped(field):
        ti.atomic_add(current_norm, field[I])
    norm_factor = norm / current_norm
    for I in ti.grouped(field):
        field[I] *= norm_factor

@ti.func
def divide_field(
    field: ti.template(),
    denom: ti.f32
):
    for I in ti.grouped(field):
        field[I] /= denom