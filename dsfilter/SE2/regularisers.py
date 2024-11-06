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
      4. `gaussian_kernel`: computes 1D Gaussian kernels using an algorithm
      based on the DIPlib[2] algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.
      5. `regularise_anisotropic`: regularise a field with a potentially
      anisotropic heat kernel, computed using a half angle approximation.[2]
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
      [2]: G. Bellaard, D.L.J. Bon, G. Pai, B.M.N. Smets, and R. Duits.
      "Analysis of (sub-)Riemannian PDE-G-CNNs". In: Journal of Mathematical
      Imaging and Vision 65 (2023), pp. 819--843.
      DOI:10.1007/s10851-023-01147-w.
"""

import taichi as ti
from dsfilter.SE2.utils import (
    mirror_spatially_on_grid
)

# Scalar Field Regularisation
## Isotropic

def gaussian_kernel(σ, truncate=5., dxy=1.):
    """Compute kernel for 1D Gaussian derivative at scale `σ`.

    Based on the DIPlib[1] algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.

    Args:
        `σ`: scale of Gaussian, taking values greater than 0.
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
    radius = int(σ * truncate / dxy + 0.5)
    k = ti.field(dtype=ti.f32, shape=2*radius+1)
    gaussian_kernel_ti(σ, radius, k)
    return k, radius

@ti.kernel
def gaussian_kernel_ti(
    σ: ti.f32,
    radius: ti.i32,
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
    normalise_field(k, 1)

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
            index = mirror_spatially_on_grid(ti.Vector([x - radius + i, y, θ], dt=ti.i32), u)
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
            index = mirror_spatially_on_grid(ti.Vector([x, y - radius + i, θ], dt=ti.i32), u)
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
            index = mirror_spatially_on_grid(ti.Vector([x, y, θ - radius + i], dt=ti.i32), u)
            s+= u[index] * k[2*radius-i]
        u_convolved[x, y, θ] = s


## Anisotropic

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
    
    Regularise `u` by convolving with a heat kernel, computed using half angle
    distance approximations.[1]

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) array to be convolved.
        `θs`: angle coordinate at each grid point.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
        `σ*`: "standard deviation" of the heat kernel in the A* direction.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) convolution
          of `u` with heat kernel.

    References:
        [1]: G. Bellaard, D.L.J. Bon, G. Pai, B.M.N. Smets, and R. Duits.
          "Analysis of (sub-)Riemannian PDE-G-CNNs". In: Journal of Mathematical
          Imaging and Vision 65 (2023), pp. 819--843.
          DOI:10.1007/s10851-023-01147-w.
    """
    # Standard deviation in pixels.
    σD1 = σ1 / dxy
    σD2 = σ2 / dxy
    σD3 = σ3 / dθ
    # Compute radii of kernel.
    truncate = 4
    rs = truncate * ti.math.ceil(ti.math.max(σD1, σD2), ti.i32)
    ro = truncate * ti.math.ceil(σD3, ti.i32)
    # Metric parameters roughly corresponding to standard deviations.
    w1 = 1/ti.math.sqrt(2 * σD1**2)
    w2 = 1/ti.math.sqrt(2 * σD2**2)
    w3 = 1/ti.math.sqrt(2 * σD3**2)
    # Group convolution definition:
    #   (K * f)(g) := ∫ Lh K(g) f(h) dμ(h) = ∫ K(h^-1) f(g h) dμ(h).
    for I in ti.grouped(u):
        # Currently at p = (x, y, θ).
        θ = θs[I]
        s = 0.
        norm = 0.
        for ix, iy, iθ in ti.ndrange(2*rs+1, 2*rs+1, 2*ro+1):
            # Evaluate kernel at q = p + (Δx, Δy, Δθ).
            Δx = (-rs + ix) * dxy
            Δy = (-rs + iy) * dxy
            Δθ = (-ro + iθ) * dθ
            # To do so, first shift q to origin with inverse of p:
            #   p^-1 q = (cos(θ) Δx + sin(θ) Δy, -sin(θ) Δx + cos(θ) Δy, Δθ).
            # Then, we find the half angle coordinates of p^-1 q:
            #   b^1 := cos(θ + Δθ/2) Δx + sin(θ + Δθ/2) Δy,
            #   b^2 := -sin(θ + Δθ/2) Δx + cos(θ + Δθ/2) Δy,
            #   b^3 := Δθ.
            b1 = ti.math.cos(θ + Δθ/2) * Δx + ti.math.sin(θ + Δθ/2) * Δy
            b2 = -ti.math.sin(θ + Δθ/2) * Δx + ti.math.cos(θ + Δθ/2) * Δy
            b3 = Δθ
            # Simple distance approximation.
            ρsq = (w1 * b1)**2 + (w2 * b2)**2 + (w3 * b3)**2
            diff = ti.math.exp(-ρsq/2)
            norm += diff
            # Actually doing a correlation, but since the kernel is symmetric,
            # i.e. K(h^-1) = K(h), this is the same.
            I_step = ti.Vector([ix - rs, iy - rs, iθ - ro], ti.i32)
            s += u[mirror_spatially_on_grid(I + I_step, u)] * diff
        u_convolved[I] = s / norm


# Matrix Field Regularisation

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
            index = mirror_spatially_on_grid(ti.Vector([x - radius + i, y, θ], dt=ti.i32), M)
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
            index = mirror_spatially_on_grid(ti.Vector([x, y - radius + i, θ], dt=ti.i32), M)
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
            index = mirror_spatially_on_grid(ti.Vector([x, y, θ - radius + i], dt=ti.i32), M)
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