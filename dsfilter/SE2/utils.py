"""
    utils
    =====

    Provides miscellaneous computational utilities that can be used with all
    controllers on SE(2).
"""

import numpy as np
import taichi as ti
from dsfilter.utils import linear_interpolate

# Preprocessing

def clean_mask_boundaries(u, mask):
    """
    Preprocesses masked data that has been lifted `u` by removing the data
    within `mask`.

    Args:
        `u`: np.ndarray(shape=(Nx, Ny, Nθ)) masked and lifted data.
        `mask`: np.ndarray(shape=(Nx, Ny, Nθ)) mask in which to remove data.
    """
    dim_K = u.shape[-1]
    u_preprocessed = np.zeros_like(u)
    median = np.median(u)
    for k in range(dim_K):
        u_preprocessed[..., k] += mask[..., k] * u[..., k] + (1 - mask[..., k]) * median
    return u_preprocessed

@ti.kernel
def project_down(
    U: ti.template(),
    u: ti.template(),
    clip_l: ti.f32,
    clip_r: ti.f32,
    scale: ti.f32,
):
    """
    @taichi.kernel

    Project orientation score `U` on SE(2) down to a function `u` on R^2 by
    integrating over orientations.

    Args:
      Static:
        `U`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) orientation score to be
          projected down.
      Mutated:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) projection of orientation
          score.
    """
    Nθ = U.shape[-1]
    for I in ti.grouped(u):
        u[I] = 0.
        for i in range(Nθ):
            u[I] += U[I, i] / scale
        u[I] = ti.math.clamp(u[I], clip_l, clip_r)

# Safe Indexing

@ti.func
def sanitize_index(
    index: ti.types.vector(3, ti.i32),
    input: ti.template()
) -> ti.types.vector(3, ti.i32):
    """
    @taichi.func
    
    Make sure the `index` is inside the shape of `input`. Copied from Gijs
    Bellaard.

    Args:
        `index`: ti.types.vector(n=3, dtype=ti.i32) index.
        `input`: ti.field in which we want to index.

    Returns:
        ti.types.vector(n=3, dtype=ti.i32) of index that is within `input`.
    """
    shape = ti.Vector(ti.static(input.shape), dt=ti.i32)
    return ti.Vector([
        ti.math.clamp(index[0], 0, shape[0] - 1),
        ti.math.clamp(index[1], 0, shape[1] - 1),
        ti.math.mod(index[2], shape[2])
    ], dt=ti.i32)

@ti.func
def mirror_spatially(
    index: ti.types.vector(3, ti.f32),
    input: ti.template()
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func
    
    Make sure the `index` is inside the shape of `input`, while reflecting at
    the boundaries.

    Args:
        `index`: ti.types.vector(n=3, dtype=ti.f32) index.
        `input`: ti.field in which we want to index.

    Returns:
        ti.types.vector(n=3, dtype=ti.f32) of index that is within `input`.
    """
    I, J, K = index
    I_max, J_max, K_max = ti.Vector(ti.static(input.shape), dt=ti.f32) - ti.Vector([1., 1., 1.], dt=ti.f32)
    return ti.Vector([
        -I * (I < 0) + I * (0 <= I <= I_max) + (2 * I_max - I) * (I > I_max),
        -J * (J < 0) + J * (0 <= J <= J_max) + (2 * J_max - J) * (J > J_max),
        ti.math.mod(K, K_max + 1)
    ], dt=ti.f32)

@ti.func
def mirror_spatially_on_grid(
    index: ti.types.vector(3, ti.i32),
    input: ti.template()
) -> ti.types.vector(3, ti.i32):
    """
    @taichi.func
    
    Make sure the `index` is inside the shape of `input`, while reflecting at
    the boundaries.

    Args:
        `index`: ti.types.vector(n=3, dtype=ti.f32) index.
        `input`: ti.field in which we want to index.

    Returns:
        ti.types.vector(n=3, dtype=ti.f32) of index that is within `input`.
    """
    I, J, K = index
    I_max, J_max, K_max = ti.Vector(ti.static(input.shape), dt=ti.i32) - ti.Vector([1, 1, 1], dt=ti.i32)
    return ti.Vector([
        -I * (I < 0) + I * (0 <= I <= I_max) + (2 * I_max - I) * (I > I_max),
        -J * (J < 0) + J * (0 <= J <= J_max) + (2 * J_max - J) * (J > J_max),
        ti.math.mod(K, K_max + 1)
    ], dt=ti.i32)

# Interpolate

@ti.func
def trilinear_interpolate(
    v000: ti.f32, 
    v001: ti.f32, 
    v010: ti.f32, 
    v011: ti.f32, 
    v100: ti.f32, 
    v101: ti.f32, 
    v110: ti.f32, 
    v111: ti.f32,
    r: ti.types.vector(3, ti.f32)
) -> ti.f32:
    """
    @taichi.func

    Interpolate value of the points `v***` depending on the distance `r`, via 
    repeated linear interpolation (x, y, θ). Adapted from Gijs Bellaard.

    Args:
        `v***`: values at points between which we want to interpolate, taking 
          real values.
        `r`: ti.types.vector(n=3, dtype=ti.f32) defining the distance to the
          points between which we to interpolate.

    Returns:
        Value of `input` interpolated at `index`.
        Interpolated value.
    """
    v00 = linear_interpolate(v000, v100, r[0])
    v01 = linear_interpolate(v001, v101, r[0])
    v10 = linear_interpolate(v010, v110, r[0])
    v11 = linear_interpolate(v011, v111, r[0])

    v0 = linear_interpolate(v00, v10, r[1])
    v1 = linear_interpolate(v01, v11, r[1])

    v = linear_interpolate(v0, v1, r[2])

    return v

@ti.func
def scalar_trilinear_interpolate(
    input: ti.template(), 
    index: ti.types.vector(3, ti.f32)
) -> ti.f32:
    """
    @taichi.func

    Interpolate value of `input` at continuous `index` trilinearly, via repeated
    linear interpolation (x, y, θ). Copied from Gijs Bellaard.

    Args:
        `input`: ti.field(dtype=[float]) in which we want to interpolate.
        `index`: ti.types.vector(n=3, dtype=ti.f32) continuous index at which we 
          want to interpolate.

    Returns:
        Value of `input` interpolated at `index`.
    """
    index_reflected = mirror_spatially(index, input)

    r = ti.math.fract(index_reflected)

    f = ti.math.floor(index_reflected, ti.i32)
    f = sanitize_index(f, input)

    c = ti.math.ceil(index_reflected, ti.i32)
    c = sanitize_index(c, input)
    
    v000 = input[f[0], f[1], f[2]]
    v001 = input[f[0], f[1], c[2]]
    v010 = input[f[0], c[1], f[2]]
    v011 = input[f[0], c[1], c[2]]
    v100 = input[c[0], f[1], f[2]]
    v101 = input[c[0], f[1], c[2]]
    v110 = input[c[0], c[1], f[2]]
    v111 = input[c[0], c[1], c[2]]

    v = trilinear_interpolate(v000, v001, v010, v011, v100, v101, v110, v111, r)

    return v

# Coordinate Transforms

@ti.func
def vectorfield_LI_to_static(
    vectorfield_LI: ti.template(),
    θs: ti.template(),
    vectorfield_static: ti.template()
):
    """
    @taichi.func

    Change the components of the vectorfield represented by `vectorfield_LI`
    from the left invariant to the static frame.

    Args:
      Static:
        `vectorfield_LI`: ti.Vector.field(n=3, dtype=[float]) represented in LI
          components.
        `θs`: angle coordinate at each grid point.
      Mutated:
        vectorfield_static`: ti.Vector.field(n=3, dtype=[float]) represented in
          static components.
    """
    for I in ti.grouped(vectorfield_LI):
        vectorfield_static[I] = vector_LI_to_static(vectorfield_LI[I], θs[I])

@ti.func
def vector_LI_to_static(
    vector_LI: ti.types.vector(3, ti.f32),
    θ: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Change the components of the vector represented by `vector_LI` from the 
    left invariant to the static frame, given that the angle coordinate of the 
    point on the manifold corresponding to this vector is θ.

    Args:
      Static:
        `vector_LI`: ti.Vector(n=3, dtype=[float]) represented in LI
          components.
        `θ`: angle coordinate of corresponding point on the manifold.
    """
    
    # A1 = [cos(θ),sin(θ),0]
    # A2 = [-sin(θ),cos(θ),0]
    # A3 = [0,0,1]

    return ti.Vector([
        ti.math.cos(θ) * vector_LI[0] - ti.math.sin(θ) * vector_LI[1],
        ti.math.sin(θ) * vector_LI[0] + ti.math.cos(θ) * vector_LI[1],
        vector_LI[2]
    ], dt=ti.f32)

@ti.func
def vectorfield_static_to_LI(
    vectorfield_static: ti.template(),
    θs: ti.template(),
    vectorfield_LI: ti.template()
):
    """
    @taichi.func

    Change the components of the vectorfield represented by `vectorfield_static`
    from the static to the left invariant frame.

    Args:
      Static:
        `vectorfield_static`: ti.Vector.field(n=3, dtype=[float]) represented in
          static components.
        `θs`: angle coordinate at each grid point.
      Mutated:
        vectorfield_LI`: ti.Vector.field(n=3, dtype=[float]) represented in
          LI components.
    """
    for I in ti.grouped(vectorfield_static):
        vectorfield_static[I] = vector_static_to_LI(vectorfield_LI[I], θs[I])

def vectorfield_static_to_LI_np(X_LI, θs):
    """
    Change the components of the vectorfield represented by `vectorfield_static`
    from the static to the left invariant frame.
    """
    X_static = np.zeros_like(X_LI)
    cos = np.cos(θs)
    sin = np.sin(θs)
    X_static[..., 0] = X_LI[..., 0] * cos - X_LI[..., 1] * sin
    X_static[..., 1] = X_LI[..., 0] * sin + X_LI[..., 1] * cos
    X_static[..., 2] = X_LI[..., 2]
    return X_static

@ti.func
def vector_static_to_LI(
    vector_static: ti.types.vector(3, ti.f32),
    θ: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func

    Change the components of the vector represented by `vector_static` from the 
    left invariant to the static frame, given that the angle coordinate of the 
    point on the manifold corresponding to this vector is θ.

    Args:
      Static:
        `vector_static`: ti.Vector(n=3, dtype=[float]) represented in static
        components.
        `θ`: angle coordinate of corresponding point on the manifold.
    """

    # A1 = [cos(θ),sin(θ),0]
    # A2 = [-sin(θ),cos(θ),0]
    # A3 = [0,0,1]

    return ti.Vector([
        ti.math.cos(θ) * vector_static[0] + ti.math.sin(θ) * vector_static[1],
        -ti.math.sin(θ) * vector_static[0] + ti.math.cos(θ) * vector_static[1],
        vector_static[2]
    ], dt=ti.f32)


def coordinate_real_to_array(x, y, θ, x_min, y_min, θ_min, dxy, dθ):
    """
    Compute the array indices (I, J, K) of the point defined by real coordinates 
    (`x`, `y`, `θ`). Can broadcast over entire arrays of real coordinates.

    Args:
        `x`: x-coordinate of the point.
        `y`: y-coordinate of the point.
        `θ`: θ-coordinate of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `θ_min`: minimum value of θ-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
    """
    I = np.rint((x - x_min) / dxy).astype(int)
    J = np.rint((y - y_min) / dxy).astype(int)
    K = np.rint((θ - θ_min) / dθ).astype(int)
    return I, J, K

@ti.func
def coordinate_real_to_array_ti(
    point: ti.types.vector(3, ti.f32),
    x_min: ti.f32,
    y_min: ti.f32,
    θ_min: ti.f32,
    dxy: ti.f32,
    dθ: ti.f32
) -> ti.types.vector(3, ti.f32):
    """
    @taichi.func
    
    Compute the array indices (I, J, K) of the point defined by real coordinates 
    `point`. Can broadcast over entire arrays of real coordinates.

    Args:
        `point`: vector of x-, y-, and θ-coordinates of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `θ_min`: minimum value of θ-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
    """
    I = (point[0] - x_min) / dxy
    J = (point[1] - y_min) / dxy
    K = (point[2] - θ_min) / dθ
    return ti.Vector([I, J, K], dt=ti.f32)

def coordinate_array_to_real(I, J, K, x_min, y_min, θ_min, dxy, dθ):
    """
    Compute the real coordinates (x, y, θ) of the point defined by array indices 
    (`I`, `J`, `K`). Can broadcast over entire arrays of array indices.

    Args:
        `I`: I index of the point.
        `J`: J index of the point.
        `K`: K index of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `θ_min`: minimum value of θ-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
        `dθ`: orientational resolution, taking values greater than 0.
    """
    x = x_min + I * dxy
    y = y_min + J * dxy
    θ = θ_min + K * dθ
    return x, y, θ

def align_to_real_axis_point(point, shape):
    """
    Express `point`, given in indices with respect to standard array convention,
    in terms of indices with respect to arrays aligned with real axes (see Notes
    for more explanation). Here, `shape` gives the shape of the array in which
    we index _after_ aligning with real axes, so [Nx, Ny, Nθ].

    Args:
        `point`: Tuple[int, int, int] describing point with respect to standard
          array indexing convention.
        `shape`: shape of array, aligned to real axes, in which we want to
          index. Note that `0 <= point[0] <= shape[1] - 1`, 
          `0 <= point[1] <= shape[0] - 1`, and `0 <= point[2] <= shape[2] - 1`.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx, Nθ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny, Nθ].

        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I x ------
            | | |    |        =>          | | |    |
            v y ------                    v v ------
                 x ->                          y ->
                 J ->                          J ->  
    """
    return point[1], shape[1] - 1 - point[0], point[2]

def align_to_real_axis_scalar_field(field):
    """
    Align `field`, given in indices with respect to standard array convention, 
    with real axes (see Notes for more explanation).

    Args:
        `field`: np.ndarray of scalar field given with respect to standard array
          convention.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx, Nθ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny, Nθ].

        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I x ------
            | | |    |        =>          | | |    |
            v y ------                    v v ------
                 x ->                          y ->
                 J ->                          J ->  
    """
    field_flipped = np.flip(field, axis=0)
    field_aligned = field_flipped.swapaxes(1, 0)
    return field_aligned

def align_to_real_axis_vector_field(vector_field):
    """
    Align `vector_field`, given in indices with respect to standard array 
    convention, with real axes (see Notes for more explanation).
    
    Args:
        `vector_field`: np.ndarray of vector field given with respect to 
          standard array convention.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx, Nθ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny, Nθ].

        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I x ------
            | | |    |        =>          | | |    |
            v y ------                    v v ------
                 x ->                          y ->
                 J ->                          J ->  
    """
    vector_field_flipped = np.flip(vector_field, axis=0)
    vector_field_aligned = vector_field_flipped.swapaxes(1, 0)
    return vector_field_aligned

def align_to_standard_array_axis_point(point, shape):
    """
    Express `point`, given in indices with respect to arrays aligned with real 
    axes, in terms of indices with respect to standard array convention, (see 
    Notes for more explanation). Here, `shape` gives the shape of the array in 
    which we index _after_ aligning with standard array convention, so
    [Ny, Nx, Nθ].

    Args:
        `point`: Tuple[int, int] describing point with respect to arrays aligned
          with real axes.
        `shape`: shape of array, with respect to standard array convention, in 
          which we want to index. Note that `0 <= point[0] <= shape[1] - 1`, 
          `0 <= point[1] <= shape[0] - 1`, and `0 <= point[2] <= shape[2] - 1`.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx, Nθ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny, Nθ].

        Alignment is achieved by first transposing and subsequently flipping the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I x ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v y ------
                 y ->                          x ->
                 J ->                          J ->  
    """
    return point[1], shape[1] - 1 - point[0], point[2]

def align_to_standard_array_axis_scalar_field(field):
    """
    Align `field`, given in indices with respect to arrays aligned with real
    axes, with respect to standard array convention (see Notes for more 
    explanation).

    Args:
        `field`: np.ndarray of scalar field given in indices with respect to
          arrays aligned with real axes.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx, Nθ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny, Nθ].

        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I x ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v y ------
                 y ->                          x ->
                 J ->                          J ->  
    """
    # field_transposed = np.transpose(field, axes=(1, 0, 2))
    field_transposed = field.swapaxes(1, 0)
    field_aligned = np.flip(field_transposed, axis=0)
    return field_aligned

def align_to_standard_array_axis_vector_field(vector_field):
    """
    Align `vector_field`, given in with respect to standard array convention, 
    with real axes (see Notes for more explanation).

    Args:
        `vector_field`: np.ndarray of vector field given in indices with respect
          to arrays aligned with real axes.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx, Nθ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny, Nθ].

        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I x ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v y ------
                 y ->                          x ->
                 J ->                          J ->  
    """
    vector_field_transposed = vector_field.swapaxes(1, 0)
    vector_field_aligned = np.flip(vector_field_transposed, axis=0)
    return vector_field_aligned