"""
    DSFilter
    ======

    The Python package *dsfilter* contains methods to apply Diffusion-Shock
    filtering, as described in Schaefer and Weickert,[1][2] on R^2 and SE(2).

    One application ...

    Summary: enhance images by applying Diffusion-Shock filtering in R^2 and
    SE(2).
    
    References:
      [1]: K. Schaefer and J. Weickert.
      "Diffusion-Shock Inpainting". In: Scale Space and Variational Methods in
      Computer Vision 14009 (2023), pp. 588--600.
      DOI:10.1137/15M1018460.
      [2]: K. Schaefer and J. Weickert.
      "Regularised Diffusion-Shock Inpainting". In: Journal of Mathematical
      Imaging and Vision (2024).
      DOI:10.1007/s10851-024-01175-0.
"""

# Access entire backend
import dsfilter.utils
import dsfilter.visualisations
import dsfilter.orientationscore
import dsfilter.R2
import dsfilter.R3
import dsfilter.SE2

# Most important functions are available at top level
## R2
from dsfilter.R2.filter import DS_filter as DS_filter_R2
## R3
from dsfilter.R3.filter import DS_filter as DS_filter_R3
## SE(2)
### Left invariant
from dsfilter.SE2.LI.filter import DS_filter as DS_filter_LI
