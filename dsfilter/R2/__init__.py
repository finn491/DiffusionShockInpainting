"""
    R2
    ==

    Apply Diffusion-Shock filtering on R^2.

    Provides the following "top level" submodule:
      1. filter: apply the Diffusion-Shock filter to an image on R^2.

    Additionally, we have the following "internal" submodules
      1. derivatives: compute various derivatives of functions on R^2.
      2. switches: compute the quantities that switch between diffusion and
      shock and between erosion and dilation.
"""

# Access entire backend
import dsfilter.R2.filter
import dsfilter.R2.derivatives
import dsfilter.R2.switches
import dsfilter.R2.utils