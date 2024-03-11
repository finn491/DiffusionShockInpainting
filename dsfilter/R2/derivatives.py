# derivativesR2.py

import taichi as ti
from dsfilter.R2.utils import sanitize_index
from dsfilter.utils import (
    select_upwind_derivative_dilation,
    select_upwind_derivative_erosion
)

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
        current_norm += field[I]
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

# Actual Derivatives

@ti.kernel
def laplacian(
    u: ti.template(),
    dxy: ti.f32,
    laplacian_u: ti.template()
):
    """
    @taichi.kernel

    Compute an approximation of the Laplacian of `u` using axial and diagonal
    central differences, as found in "Diffusion-Shock Inpainting" (2023) by K.
    Schaefer and J. Weickert, Eq. (9).

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `laplacian_u`: ti.field(dtype=[float], shape=shape) of laplacian of `u`,
          which is updated in place.
    """
    δ = ti.math.sqrt(2) - 1 # Good value for rotation invariance according to M. Welk and J. Weickert (2021)
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
        # Axial Stencil
        # 0 |  1 | 0
        # 1 | -4 | 1
        # 0 |  1 | 0
        laplacian_u[I] = (1 - δ) / dxy**2 * (
            -4 * u[I] +
            u[I_dx_forward] +
            u[I_dx_backward] +
            u[I_dy_forward] +
            u[I_dy_backward]
        )
        # Diagonal Stencil
        # 1 |  0 | 1
        # 0 | -4 | 0
        # 1 |  0 | 1
        laplacian_u[I] += δ / (2 * dxy**2) * (
            -4 * u[I] +
            u[I_dplus_forward] +
            u[I_dplus_backward] +
            u[I_dminus_forward] +
            u[I_dminus_backward]
        )

@ti.kernel
def morphological(
    u: ti.template(),
    dxy: ti.f32,
    dilation_u: ti.template(),
    erosion_u: ti.template()
):
    """
    @taichi.kernel

    Compute an approximation of the |grad `u`| using axial and diagonal upwind
    differences, as found in "Diffusion-Shock Inpainting" (2023) by K.
    Schaefer and J. Weickert, Eq. (12).

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=shape) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `dilation_u`: ti.field(dtype=[float], shape=shape) of |grad `u`|,
          which is updated in place.
        `erosion_u`: ti.field(dtype=[float], shape=shape) of -|grad `u`|,
          which is updated in place.
    """
    δ = ti.math.sqrt(2) - 1 # Good value for rotation invariance according to M. Welk and J. Weickert (2021)
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

        d_dx_forward = (u[I_dx_forward] - u[I])
        d_dx_backward = (u[I] - u[I_dx_backward])
        d_dy_forward = (u[I_dy_forward] - u[I])
        d_dy_backward = (u[I] - u[I_dy_backward])
        d_dplus_forward = (u[I_dplus_forward] - u[I])
        d_dplus_backward = (u[I] - u[I_dplus_backward])
        d_dminus_forward = (u[I_dminus_forward] - u[I])
        d_dminus_backward = (u[I] - u[I_dminus_backward])

        # Dilation
        ## Axial
        dilation_u[I] = (1 - δ) / dxy * (
            select_upwind_derivative_dilation(d_dx_forward, d_dx_backward)**2 +
            select_upwind_derivative_dilation(d_dy_forward, d_dy_backward)**2
        )
        ## Diagonal
        dilation_u[I] += δ / (ti.math.sqrt(2) * dxy) * (
            select_upwind_derivative_dilation(d_dplus_forward, d_dplus_backward)**2 +
            select_upwind_derivative_dilation(d_dminus_forward, d_dminus_backward)**2
        )

        # Erosion
        ## Axial
        erosion_u[I] = -(1 - δ) / dxy * (
            select_upwind_derivative_erosion(d_dx_forward, d_dx_backward)**2 +
            select_upwind_derivative_erosion(d_dy_forward, d_dy_backward)**2
        )
        ## Diagonal
        erosion_u[I] -= δ / (ti.math.sqrt(2) * dxy) * (
            select_upwind_derivative_erosion(d_dplus_forward, d_dplus_backward)**2 +
            select_upwind_derivative_erosion(d_dminus_forward, d_dminus_backward)**2
        )


# @ti.func
# def abs_derivatives(
#     u: ti.template(),
#     dxy: ti.f32,
#     dx_forward: ti.template(),
#     dx_backward: ti.template(),
#     dy_forward: ti.template(),
#     dy_backward: ti.template(),
#     dplus_forward: ti.template(),
#     dplus_backward: ti.template(),
#     dminus_forward: ti.template(),
#     dminus_backward: ti.template(),
#     abs_dx_dilation: ti.template(),
#     abs_dy_dilation: ti.template(),
#     abs_dplus_dilation: ti.template(),
#     abs_dminus_dilation: ti.template(),
#     abs_dx_erosion: ti.template(),
#     abs_dy_erosion: ti.template(),
#     abs_dplus_erosion: ti.template(),
#     abs_dminus_erosion: ti.template()
# ):
#     """
#     @taichi.func

#     Compute an approximation of the absolute value of the derivative of `u` in 
#     the `x`, `y`, and diagonal directions. Adapted from Gijs.

#     Args:
#       Static:
#         `u`: ti.field(dtype=[float], shape=shape) which we want to 
#           differentiate.
#         `dxy`: step size in x and y direction, taking values greater than 0.
#       Mutated:
#         `d*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
#           updated in place.
#         `abs_d*`: ti.field(dtype=[float], shape=shape) of upwind derivatives,
#           which are updated in place.
#     """
#     derivatives(u, dxy, dx_forward, dx_backward, dy_forward, dy_backward, dplus_forward, dplus_backward, dminus_forward, 
#                 dminus_backward)
#     for I in ti.grouped(u):
#         # Axial
#         abs_dx_dilation[I] = ti.math.max(dx_forward[I], -dx_backward[I], 0)
#         abs_dy_dilation[I] = ti.math.max(dy_forward[I], -dy_backward[I], 0)
#         # Diagonal
#         abs_dplus_dilation[I] = ti.math.max(dplus_forward[I], -dplus_backward[I], 0)
#         abs_dminus_dilation[I] = ti.math.max(dminus_forward[I], -dminus_backward[I], 0)

#         # Axial
#         abs_dx_erosion[I] = ti.math.max(-dx_forward[I], dx_backward[I], 0)
#         abs_dy_erosion[I] = ti.math.max(-dy_forward[I], dy_backward[I], 0)
#         # Diagonal
#         abs_dplus_erosion[I] = ti.math.max(-dplus_forward[I], dplus_backward[I], 0)
#         abs_dminus_erosion[I] = ti.math.max(-dminus_forward[I], dminus_backward[I], 0)

# @ti.func
# def derivatives(
#     u: ti.template(),
#     dxy: ti.f32,
#     dx_forward: ti.template(),
#     dx_backward: ti.template(),
#     dy_forward: ti.template(),
#     dy_backward: ti.template(),
#     dplus_forward: ti.template(),
#     dplus_backward: ti.template(),
#     dminus_forward: ti.template(),
#     dminus_backward: ti.template()
# ):
#     """
#     @taichi.func

#     Compute the forward and backward finite differences of `u` with step size 
#     `dxy`.

#     Args:
#       Static:
#         `u`: ti.field(dtype=[float], shape=shape) which we want to 
#           differentiate.
#         `dxy`: step size in x and y direction, taking values greater than 0.
#       Mutated:
#         `d*_*`: ti.field(dtype=[float], shape=shape) of derivatives, which are 
#           updated in place.
#     """
#     I_dx = ti.Vector([1, 0], dt=ti.i32)
#     I_dy = ti.Vector([0, 1], dt=ti.i32)
#     I_dplus = I_dx + I_dy  # Positive diagonal
#     I_dminus = I_dx - I_dy # Negative diagonal
#     for I in ti.grouped(u):
#         # We do not need to interpolate because we always end up on the grid.
#         # Axial
#         I_dx_forward = sanitize_index(I + I_dx, u)
#         I_dx_backward = sanitize_index(I - I_dx, u)
#         I_dy_forward = sanitize_index(I + I_dy, u)
#         I_dy_backward = sanitize_index(I - I_dy, u)
#         dx_forward[I] = (u[I_dx_forward] - u[I]) / dxy
#         dx_backward[I] = (u[I] - u[I_dx_backward]) / dxy
#         dy_forward[I] = (u[I_dy_forward] - u[I]) / dxy
#         dy_backward[I] = (u[I] - u[I_dy_backward]) / dxy
#         # Diagonal
#         I_dplus_forward = sanitize_index(I + I_dplus, u)
#         I_dplus_backward = sanitize_index(I - I_dplus, u)
#         I_dminus_forward = sanitize_index(I + I_dminus, u)
#         I_dminus_backward = sanitize_index(I - I_dminus, u)
#         dplus_forward[I] = (u[I_dplus_forward] - u[I]) / (ti.math.sqrt(2) * dxy)
#         dplus_backward[I] = (u[I] - u[I_dplus_backward]) / (ti.math.sqrt(2) * dxy)
#         dminus_forward[I] = (u[I_dminus_forward] - u[I]) / (ti.math.sqrt(2) * dxy)
#         dminus_backward[I] = (u[I] - u[I_dminus_backward]) / (ti.math.sqrt(2) * dxy)

@ti.func
def central_derivatives_second_order(
    u: ti.template(),
    dxy: ti.f32,
    d_dxx: ti.template(),
    d_dxy: ti.template(),
    d_dyy: ti.template()
):
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

        d_dxx[I] = (
            u[I_dx_forward] -
            u[I] * 2 +
            u[I_dx_backward]
        ) / dxy**2

        d_dxy[I] = (
            u[I_dplus_forward] -
            u[I_dminus_forward] -
            u[I_dminus_backward] +
            u[I_dplus_backward]
        ) / (4* dxy**2)

        d_dyy[I] = (
            u[I_dy_forward] -
            u[I] * 2 +
            u[I_dy_backward]
        ) / dxy**2

# Gaussian derivatives

# We cannot nest parallelised loops in if-else statements in TaiChi kernels.

@ti.func
def convolve_with_kernel_x_dir(
    u_padded: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    u_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve `u_padded` the 1D kernel `k` in the x-direction.

    Args:
      Static:
        `u_padded`: ti.field(dtype=ti.f32, shape=shape_padded) of array to be
          convolved, with shape_padded[i] = shape[i] + 2 * `radius`.
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=ti.f32, shape=shape) of convolution of 
          `u_padded` with `k`.
    """
    for x, y in u_convolved:
        y_shifted = y + radius
        s = 0.
        for i in range(2*radius+1):
            s += u_padded[x + i, y_shifted] * k[2*radius+1-i]
        u_convolved[x, y] = s

@ti.func
def convolve_with_kernel_y_dir(
    u_padded: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    u_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve `u_padded` the 1D kernel `k` in the y-direction.

    Args:
      Static:
        `u_padded`: ti.field(dtype=ti.f32, shape=shape_padded) of array to be
          convolved, with shape_padded[i] = shape[i] + 2 * `radius`.
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=ti.f32, shape=shape) of convolution of 
          `u_padded` with `k`.
    """
    for x, y in u_convolved:
        x_shifted = x + radius
        s = 0.
        for i in range(2*radius+1):
            s+= u_padded[x_shifted, y + i] * k[2*radius+1-i]
        u_convolved[x, y] = s

def gaussian_derivative_kernel(σ, order, truncate=5., dxy=1.):
    """Compute kernel for 1D Gaussian derivative of order `order` at scale `σ`.

    Based on the DIPlib algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.

    Args:
        `σ`: scale of Gaussian, taking values greater than 0.
        `order`: order of the derivative, taking values 0 or 1.
        `truncate`: number of scales `σ` at which kernel is truncated, taking 
          values greater than 0.
        `dxy`: step size in x and y direction, taking values greater than 0.

    Returns:
        Tuple ti.field(dtype=ti.f32, shape=2*radius+1) of the Gaussian kernel
          and the radius of the kernel.
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

    Based on the DIPlib algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.

    Args:
      Static:
        `σ`: scale of Gaussian, taking values greater than 0.
        `radius`: radius at which kernel is truncated, taking integer values
          greater than 0.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) of kernel, which is
          updated in place.
    """
    ti.loop_config(serialize=True)
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

    Based on the DIPlib algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.

    Args:
      Static:
        `σ`: scale of Gaussian, taking values greater than 0.
        `radius`: radius at which kernel is truncated, taking integer values
          greater than 0.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) of kernel, which is
          updated in place.
    """
    moment = 0.
    ti.loop_config(serialize=True)
    for i in range(2*radius+1):
        x = -radius + i
        val = x * ti.math.exp(-x**2 / (2 * σ**2))
        moment += x * val
        k[i] = val
    divide_field(k, -moment * dxy)