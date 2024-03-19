# derivativesR2.py

import taichi as ti
from dsfilter.R2.utils import sanitize_index
from dsfilter.utils import (
    select_upwind_derivative_dilation,
    select_upwind_derivative_erosion
)
# Actual Derivatives

@ti.kernel
def laplacian(
    u_padded: ti.template(),
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
        `u_padded`: ti.field(dtype=[float], shape=[Nx+2, Ny+2]) u padded with
          reflecting boundaries.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `laplacian_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of laplacian of
          u, which is updated in place.
    """
    δ = ti.math.sqrt(2) - 1 # Good value for rotation invariance according to M. Welk and J. Weickert (2021)
    I_shift = ti.Vector([1, 1], dt=ti.i32)
    I_dx = ti.Vector([1, 0], dt=ti.i32)
    I_dy = ti.Vector([0, 1], dt=ti.i32)
    I_dplus = I_dx + I_dy  # Positive diagonal
    I_dminus = I_dx - I_dy # Negative diagonal
    for I_unshifted in ti.grouped(laplacian_u):
        I = I_unshifted + I_shift # Account for the padding.
        # Axial Stencil
        # 0 |  1 | 0
        # 1 | -4 | 1
        # 0 |  1 | 0
        laplacian_u[I_unshifted] = (1 - δ) / dxy**2 * (
            -4 * u_padded[I] +
            u_padded[I + I_dx] +
            u_padded[I - I_dx] +
            u_padded[I + I_dy] +
            u_padded[I - I_dy]
        )
        # Diagonal Stencil
        # 1 |  0 | 1
        # 0 | -4 | 0
        # 1 |  0 | 1
        laplacian_u[I_unshifted] += δ / (2 * dxy**2) * (
            -4 * u_padded[I] +
            u_padded[I + I_dplus] +
            u_padded[I - I_dplus] +
            u_padded[I + I_dminus] +
            u_padded[I - I_dminus]
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

    Compute an approximation of the ||grad `u`|| using axial and diagonal upwind
    differences, as found in "Diffusion-Shock Inpainting" (2023) by K.
    Schaefer and J. Weickert, Eq. (12).

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx+2, Ny+2]) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `dilation_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of ||grad `u`||,
          which is updated in place.
        `erosion_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of -||grad `u`||,
          which is updated in place.
    """
    δ = ti.math.sqrt(2) - 1 # Good value for rotation invariance according to M. Welk and J. Weickert (2021)
    I_shift = ti.Vector([1, 1], dt=ti.i32)
    I_dx = ti.Vector([1, 0], dt=ti.i32)
    I_dy = ti.Vector([0, 1], dt=ti.i32)
    I_dplus = I_dx + I_dy  # Positive diagonal
    I_dminus = I_dx - I_dy # Negative diagonal
    for I_unshifted in ti.grouped(dilation_u):
        I = I_unshifted + I_shift # Account for the padding.

        d_dx_forward = u[I + I_dx] - u[I]
        d_dx_backward = u[I] - u[I - I_dx]
        d_dy_forward = u[I + I_dy] - u[I]
        d_dy_backward = u[I] - u[I - I_dy]
        d_dplus_forward = u[I + I_dplus] - u[I]
        d_dplus_backward = u[I] - u[I - I_dplus]
        d_dminus_forward = u[I + I_dminus] - u[I]
        d_dminus_backward = u[I] - u[I - I_dminus]

        # Dilation
        ## Axial
        dilation_u[I_unshifted] = (1 - δ) / dxy * ti.math.sqrt(
            select_upwind_derivative_dilation(d_dx_forward, d_dx_backward)**2 +
            select_upwind_derivative_dilation(d_dy_forward, d_dy_backward)**2
        )
        ## Diagonal
        dilation_u[I_unshifted] += δ / (ti.math.sqrt(2) * dxy) * ti.math.sqrt(
            select_upwind_derivative_dilation(d_dplus_forward, d_dplus_backward)**2 +
            select_upwind_derivative_dilation(d_dminus_forward, d_dminus_backward)**2
        )

        # Erosion
        ## Axial
        erosion_u[I_unshifted] = -(1 - δ) / dxy * ti.math.sqrt(
            select_upwind_derivative_erosion(d_dx_forward, d_dx_backward)**2 +
            select_upwind_derivative_erosion(d_dy_forward, d_dy_backward)**2
        )
        ## Diagonal
        erosion_u[I_unshifted] -= δ / (ti.math.sqrt(2) * dxy) * ti.math.sqrt(
            select_upwind_derivative_erosion(d_dplus_forward, d_dplus_backward)**2 +
            select_upwind_derivative_erosion(d_dminus_forward, d_dminus_backward)**2
        )

@ti.func
def central_derivatives_second_order(
    u: ti.template(),
    dxy: ti.f32,
    d_dxx: ti.template(),
    d_dxy: ti.template(),
    d_dyy: ti.template()
):
    """
    @taichi.func

    Compute the second order derivatives of `u` using central differences.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `d_d**`: ti.field(dtype=[float], shape=[Nx, Ny]) of d* d* `u`, which is
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
        `u_padded`: ti.field(dtype=[float], shape=[Nx+2*`radius`, Ny]) array to
          be convolved.
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny]) of convolution of 
          `u_padded` with `k`.
    """
    for x, y in u_convolved:
        y_shifted = y
        s = 0.
        for i in range(2*radius+1):
            s += u_padded[x + i, y_shifted] * k[2*radius-i]
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
        `u_padded`: ti.field(dtype=[float], shape=[Nx, Ny+2*`radius`]) array to
          be convolved.
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny]) of convolution of 
          `u_padded` with `k`.
    """
    for x, y in u_convolved:
        x_shifted = x
        s = 0.
        for i in range(2*radius+1):
            s+= u_padded[x_shifted, y + i] * k[2*radius-i]
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
        Tuple ti.field(dtype=[float], shape=2*radius+1) of the Gaussian kernel
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
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel, which is
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
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel, which is
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