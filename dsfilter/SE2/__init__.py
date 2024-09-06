"""
    SE2
    ===

    Apply Diffusion-Shock filtering on SE(2).

    Provides the following "top level" submodule:
      1. LI: perform DS filtering on SE(2) using left invariant vector fields.
      2. TODO: gauge: perform DS filtering on SE(2) using gauge frames.

    Additionally, we have the following "internal" submodules
      1. regularisers: 
      2. utils: 
"""

# Access entire backend
import dsfilter.SE2.regularisers
import dsfilter.SE2.utils
import dsfilter.SE2.LI
import dsfilter.SE2.gauge