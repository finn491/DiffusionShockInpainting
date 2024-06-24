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
      3. regularisers:
      4. utils:
"""

# Access entire backend
import dsfilter.R3.filter
import dsfilter.R3.derivatives
import dsfilter.R3.switches
import dsfilter.R3.regularisers
import dsfilter.R3.utils