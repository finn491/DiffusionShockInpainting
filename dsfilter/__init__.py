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
      "Regularised Diffusion-Shock Inpainting". arXiv preprint. 
      DOI:10.48550/arXiv.2309.08761.
"""

# Access entire backend
import dsfilter.utils
import dsfilter.visualisations
import dsfilter.R2

# Most important functions are available at top level
## R2
from dsfilter.R2.filter import DS_filter as DS_filter_R2
## SE(2)
### Left invariant
from dsfilter.SE2.LI.filter import DS_filter as DS_filter_LI
