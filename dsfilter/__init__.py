"""
    DSFilter
    ======

    The Python package *dsfilter* contains methods to apply Diffusion-Shock
    filtering, as described in Schaefer and Weickert "Diffusion-Shock
    Inpainting" (2023), on R^2 and SE(2).

    One application ...

    Summary: enhance images by applying Diffusion-Shock filtering in R^2 and
    SE(2).
"""

# Access entire backend
import dsfilter.utils
import dsfilter.R2

# Most important functions are available at top level
from dsfilter.R2.filter import DS_filter_R2
