"""
    metric
    ======

    Provides tools to deal with metrics R^2. The primary methods are:
      1. `invert_metric`: compute the matrix defining the dual metric from the
      matrix defining the primal metric.
      2. `normalise`: normalise a vector to norm 1 with respect to some
      data-driven metric.
      3. `norm`: compute the norm of a vector with respect to some data-driven
      metric.
"""

import taichi as ti

# Metric Inversion

def invert_metric(G):
    """
    Invert the metric tensor defined by the matrix `G`.

    Args:
        `G`: np.ndarray(shape=(2,), dtype=[float]) of constants of the diagonal
          metric tensor with respect to standard basis. Defaults to standard
          Euclidean metric.
    """
    G_inv = 1 / G
    return G_inv

# Normalisation

@ti.func
def normalise(
    vec: ti.types.vector(2, ti.f32),
    G: ti.types.vector(2, ti.f32),
    cost: ti.f32
) -> ti.types.vector(2, ti.f32):
    """
    @taichi.func

    Normalise `vec`, represented in standard coordinates, to 1 with respect to 
    the datadriven left invariant metric tensor defined by `G` and `cost`.

    Args:
        `vec`: ti.types.vector(n=2, dtype=[float]) which we want to normalise.
        `G`: ti.types.vector(n=2, dtype=[float]) of constants of the diagonal
          metric tensor with respect to standard basis.
        `cost`: cost function at point, taking values between 0 and 1.

    Returns:
        ti.types.vector(n=2, dtype=[float]) of normalisation of `vec`.
    """
    return vec / norm(vec, G, cost)

@ti.func
def norm(
    vec: ti.types.vector(2, ti.f32),
    G: ti.types.vector(2, ti.f32),
    cost: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Compute the norm of `vec` represented in standard coordinates with respect
    to the left invariant metric tensor defined by `G`.

    Args:
        `vec`: ti.types.vector(n=2, dtype=[float]) which we want to normalise.
        `G`: ti.types.vector(n=2, dtype=[float]) of constants of the diagonal
          metric tensor with respect to standard basis.
        `cost`: cost function at point, taking values between 0 and 1.

    Returns:
        Norm of `vec`.
    """
    c_1, c_2 = vec[0], vec[1]
    return ti.math.sqrt(
            G[0] * c_1**2 +
            G[1] * c_2**2
    ) * cost