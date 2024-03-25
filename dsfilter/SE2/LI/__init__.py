"""
    SE2
    ==

    Apply Diffusion-Shock filtering on SE(2) using left invariant vector fields.

    Provides the following "top level" submodule:
      1. filter: apply the Diffusion-Shock filter to an image on SE(2).

    Additionally, we have the following "internal" submodules
      1. derivatives: compute various derivatives of functions on SE(2).
      2. switches: compute the quantities that switch between diffusion and
      shock and between erosion and dilation.
"""

# Access entire backend
import dsfilter.SE2.LI.filter
import dsfilter.SE2.LI.derivatives
import dsfilter.SE2.LI.switches