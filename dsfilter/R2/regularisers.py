"""
    regularisers
    ============

    Provides tools to regularise scalar fields on R^2, namely:
      1. `convolve_with_kernel_x_dir`: convolve a field with a 1D kernel along
      the x-direction.
      2. `convolve_with_kernel_y_dir`: convolve a field with a 1D kernel along
      the y-direction.
      3. `gaussian_derivative_kernel`: computes 1D Gaussian derivative kernels
      of order 0 and 1, using an algorithm that improves the accuracy of higher
      order derivative kernels with small widths, based on the DIPlib[1]
      algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.

    References:
      [1]: C. Luengo, W. Caarls, R. Ligteringen, E. Schuitema, Y. Guo,
      E. Wernersson, F. Malmberg, S. Lokhorst, M. Wolff, G. van Kempen,
      M. van Ginkel, L. van Vliet, B. Rieger, B. Verwer, H. Netten,
      J. W. Brandenburg, J. Dijk, N. van den Brink, F. Faas, K. van Wijk,
      and T. Pham. "DIPlib 3". GitHub: https://github.com/DIPlib/diplib.
"""

import taichi as ti
from dsfilter.R2.utils import sanitize_reflected_index

# We cannot nest parallelised loops in if-else statements in TaiChi kernels.

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
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) array to
          be convolved.
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny]) convolution of
        `u` with `k`.
    """
    for x, y in u_convolved:
        s = 0.
        for i in range(2*radius+1):
            index = sanitize_reflected_index(ti.Vector([x - radius + i, y], dt=ti.i32), u)
            s += u[index] * k[2*radius-i]
        u_convolved[x, y] = s

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
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) array to be convolved.
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny]) convolution of 
          `u` with `k`.
    """
    for x, y in u_convolved:
        s = 0.
        for i in range(2*radius+1):
            index = sanitize_reflected_index(ti.Vector([x, y - radius + i], dt=ti.i32), u)
            s+= u[index] * k[2*radius-i]
        u_convolved[x, y] = s

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
    normalise_field(k, 1/dxy)

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