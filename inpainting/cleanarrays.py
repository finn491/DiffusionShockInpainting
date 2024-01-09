# cleanarrays.py

import taichi as ti
import numpy as np
import diplib as dip
from PIL import Image
import matplotlib.pyplot as plt

# NumPy Arrays


def pad_array(u, pad_value=0., pad_shape=1):
    """
    Pad a numpy array `u` in each direction with `pad_shape` entries of values 
    `pad_value`.

    Args:
        `u`: np.ndarray that is to be padded.
        `pad_value`: Value with which `u` is padded.
        `pad_shape`: Number of cells that is padded in each direction, taking
          positive integral values.

    Returns:
        np.ndarray of the padded array.
    """
    return np.pad(u, pad_width=pad_shape, constant_values=pad_value)


def extract_centre_slice(u, pad_shape=1):
    """
    Select the centre part of `u`, removing padding of size `pad_shape` in each 
    direction.

    Args:
        `u`: np.ndarray from which the centre is to be extracted.
        `pad_shape`: Number of cells that is padded in each direction.

    Returns:
        np.ndarray of the unpadded array.
    """
    if type(pad_shape) == int:
        centre_slice = tuple(slice(pad_shape, dim_len - pad_shape, 1) for dim_len in u.shape)
    else: 
        centre_slice = tuple(slice(pad_shape[i], dim_len - pad_shape[i], 1) for i, dim_len in enumerate(u.shape))
    return centre_slice


def unpad_array(u, pad_shape=1):
    """
    Remove the outer `pad_shape` entries numpy array `u`.

    Args:
        `u`: np.ndarray from which the centre is to be extracted.
        `pad_shape`: Number of cells that is padded in each direction, taking
          positive integral values.

    Returns:
        np.ndarray of the unpadded array. Note that 
        `unpad_array(*, pad_shape=pad_shape)` is the left-inverse of
        `pad_array(*, pad_value=pad_value, pad_shape=pad_shape)` for all
        `pad_value`, `pad_shape`.
    """
    centre_slice = extract_centre_slice(u, pad_shape=pad_shape)
    return u[centre_slice]

def convert_array_to_image(image_array):
    """Convert numpy array `image_array` to a grayscale PIL Image object."""
    if image_array.dtype == "uint8":
        image = Image.fromarray(image_array, mode="L")
    else:
        image = Image.fromarray((image_array * 255).astype("uint8"), mode="L")
    return image

def image_rescale(image_array, new_max=1.):
    """
    Affinely rescale values in numpy array `image_array` to be between 0. and
    `new_max`.
    """
    image_max = image_array.max()
    image_min = image_array.min()
    return new_max * (image_array - image_min) / (image_max - image_min)

def high_pass_filter(image_array, σs):
    """Apply a high pass filter with Gaussian scales `σs` to `image_array`."""
    low_frequencies = dip.Gauss(image_array, σs)
    image_array_unnormalised = image_array - low_frequencies
    image_array_filtered = image_rescale(image_array_unnormalised)
    return image_array_filtered

def view_image_array(image_array):
    """View numpy array `image_array` as a grayscale image."""
    image = convert_array_to_image(image_array)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_axis_off()
    return image, fig, ax

def view_image_arrays_side_by_side(image_array_list):
    """
    View list of numpy array `image_array_list` side by side as grayscale 
    images.
    """
    image_list = []
    ncols = len(image_array_list)
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(5 * ncols, 5))
    for i, image_array in enumerate(image_array_list):
        image = convert_array_to_image(image_array)
        image_list.append(image)
        ax[i].imshow(image_array, cmap="gray", origin="upper")
        ax[i].set_axis_off()
    return image_list, fig, ax

# TaiChi Fields


@ti.kernel
def apply_boundary_conditions(
    u: ti.template(), 
    boundarypoints: ti.template(), 
    boundaryvalues: ti.template()
):
    """
    @taichi.kernel

    Apply `boundaryvalues` at `boundarypoints` to `u`.

    Args:
        `u`: ti.field(dtype=[float]) to which the boundary conditions should be 
          applied.
        `boundarypoints`: ti.Vector.field(n=dim, dtype=[int], shape=N_points),
          where `N_points` is the number of boundary points and `dim` is the 
          dimension of `u`.
        `boundaryvalues`: ti.Vector.field(n=dim, dtype=[float], shape=N_points),
          where `N_points` is the number of boundary points and `dim` is the 
          dimension of `u`.

    Returns:
        ti.field equal to `u`, except at `boundarypoints`, where it equals
          `boundaryvalues`.
    """
    for I in ti.grouped(boundarypoints):
        u[boundarypoints[I]] = boundaryvalues[I]
