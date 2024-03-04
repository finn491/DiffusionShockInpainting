"""
    visualisations
    ==============
    
    Provides methods to visualise 2D and 3D images using matplotlib.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dsfilter.R2.utils import (
    align_to_standard_array_axis_scalar_field,
    align_to_standard_array_axis_vector_field
)
# from dsfilter.SO3.utils import align_to_standard_array_axis_scalar_field as align_to_standard_array_axis_scalar_field_SO3


def convert_array_to_image(image_array):
    """Convert numpy array `image_array` to a grayscale PIL Image object."""
    image_array_aligned = align_to_standard_array_axis_scalar_field(image_array)
    
    if image_array_aligned.dtype == "uint8":
        image = Image.fromarray(image_array_aligned, mode="L")
    else:
        image = Image.fromarray((image_array_aligned * 255).astype("uint8"), mode="L")
    return image

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

def plot_image_array(image_array, x_min, x_max, y_min, y_max, cmap="gray", figsize=(10, 10), fig=None, ax=None):
    """Plot `image_array` as a heatmap."""
    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    image_array_aligned = align_to_standard_array_axis_scalar_field(image_array)

    ax.imshow(image_array_aligned, cmap=cmap, extent=(x_min, x_max, y_min, y_max))
    return fig, ax

# def plot_image_array_SO3(image_array, α_min, α_max, β_min, β_max, cmap="gray", figsize=(10, 10), fig=None, ax=None):
#     """Plot `image_array` as a heatmap."""
#     if fig is None and ax is None:
#         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#         ax.set_xlabel("$β$")
#         ax.set_ylabel("$α$")
#         ax.set_xlim(β_max, β_min)
#         ax.set_ylim(α_min, α_max)

#     image_array_aligned = align_to_standard_array_axis_scalar_field_SO3(np.expand_dims(image_array, axis=2)).squeeze(-1)

#     ax.imshow(image_array_aligned, cmap=cmap, extent=(β_max, β_min, α_min, α_max))
#     return fig, ax

def plot_contour(distance, xs, ys, levels=None, linestyles=None, figsize=(12, 10), fig=None, ax=None, x_min=None, 
                 x_max=None, y_min=None, y_max=None):
    """Plot the contours of the two-dimensional array `distance`."""
    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        x_min = overwrite_default(x_min, xs[0, 0])
        x_max = overwrite_default(x_max, xs[-1, -1])
        y_min = overwrite_default(y_min, ys[0, 0])
        y_max = overwrite_default(y_max, ys[-1, -1])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    xs_aligned = align_to_standard_array_axis_scalar_field(xs)
    ys_aligned = align_to_standard_array_axis_scalar_field(ys)
    distance_aligned = align_to_standard_array_axis_scalar_field(distance)
    
    contour = ax.contour(xs_aligned, ys_aligned, distance_aligned, levels=levels, linestyles=linestyles)
    return fig, ax, contour

# def plot_contour_SO3(distance, αs, βs, levels=None, linestyles=None, figsize=(12, 10), fig=None, ax=None, α_min=None,
#                      α_max=None, β_min=None, β_max=None):
#     """Plot the contours of the two-dimensional array `distance`."""
#     if fig is None and ax is None:
#         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
#         ax.set_xlabel("$β$")
#         ax.set_ylabel("$α$")
#         α_min = overwrite_default(α_min, αs[0, 0])
#         α_max = overwrite_default(α_max, αs[-1, -1])
#         β_min = overwrite_default(β_min, βs[0, 0])
#         β_max = overwrite_default(β_max, βs[-1, -1])
#         ax.set_xlim(β_max, β_min)
#         ax.set_ylim(α_min, α_max)

#     αs_aligned = align_to_standard_array_axis_scalar_field_SO3(αs)
#     βs_aligned = align_to_standard_array_axis_scalar_field_SO3(βs)
#     distance_aligned = align_to_standard_array_axis_scalar_field_SO3(np.expand_dims(distance, axis=2)).squeeze(-1)
    
#     contour = ax.contour(βs_aligned, αs_aligned, distance_aligned, levels=levels, linestyles=linestyles)
#     return fig, ax, contour

def plot_isosurface(verts, faces, x_min, x_max, y_min, y_max, θ_min, θ_max, dxy, dθ, alpha=0.5, label=None,
                    figsize=(10, 10), fig=None, ax=None):
    """Plot the isosurface given by `verts` and `faces`."""
    if fig is None and ax is None:
        fig = plt.figure(figsize=figsize)
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.resizable = False
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(θ_min, θ_max)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$θ$")
    ax.plot_trisurf(x_min + verts[:, 1] * dxy, y_max - verts[:, 0] * dxy, faces, θ_min + verts[:, 2] * dθ, alpha=alpha,
                    label=label)
    return fig, ax

# def plot_isosurface_SO3(verts, faces, α_min, α_max, β_min, β_max, φ_min, φ_max, dα, dβ, dφ, alpha=0.5, label=None,
#                     figsize=(10, 10), fig=None, ax=None):
#     """Plot the isosurface given by `verts` and `faces`."""
#     if fig is None and ax is None:
#         fig = plt.figure(figsize=figsize)
#         fig.canvas.toolbar_visible = False
#         fig.canvas.header_visible = False
#         fig.canvas.footer_visible = False
#         fig.canvas.resizable = False
#         ax = fig.add_subplot(111, projection="3d")
#         ax.set_xlim(β_max, β_min)
#         ax.set_ylim(α_min, α_max)
#         ax.set_zlim(φ_min, φ_max)
#         ax.set_xlabel("$β$")
#         ax.set_ylabel("$α$")
#         ax.set_zlabel("$φ$")
#     ax.plot_trisurf(β_min + verts[:, 0] * dβ, α_min + verts[:, 1] * dα, faces, φ_min + verts[:, 2] * dφ, alpha=alpha,
#                     label=label)
#     return fig, ax

def overwrite_default(passed_value, default_value):
    """
    Overwrite the value of some parameter if the user has not passed that
    parameter.
    """
    if passed_value is None: # User did not pass any value
        passed_value = default_value
    return default_value

def plot_vector_field(vector_field, xs, ys, color="red", figsize=(10, 10), fig=None, ax=None, x_min=None, x_max=None, 
                      y_min=None, y_max=None):
    """Streamplot of `vector_field`."""
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        x_min = overwrite_default(x_min, xs[0, 0])
        x_max = overwrite_default(x_max, xs[-1, -1])
        y_min = overwrite_default(y_min, ys[0, 0])
        y_max = overwrite_default(y_max, ys[-1, -1])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    xs_aligned = align_to_standard_array_axis_scalar_field(xs)
    ys_aligned = align_to_standard_array_axis_scalar_field(ys)
    vector_field_aligned = align_to_standard_array_axis_vector_field(vector_field)

    ax.streamplot(np.flip(xs_aligned, axis=0), np.flip(ys_aligned, axis=0),
                  np.flip(vector_field_aligned[..., 0], axis=0), 
                  np.flip(vector_field_aligned[..., 1], axis=0), color=color)
    return fig, ax

def plot_scalar_field(scalar_field, xs, ys, levels=None, figsize=(12, 10), fig=None, ax=None, x_min=None, x_max=None, 
                      y_min=None, y_max=None):
    """Plot two-dimensional `scalar_field` using coloured in contours."""
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        x_min = overwrite_default(x_min, xs[0, 0])
        x_max = overwrite_default(x_max, xs[-1, -1])
        y_min = overwrite_default(y_min, ys[0, 0])
        y_max = overwrite_default(y_max, ys[-1, -1])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    xs_aligned = align_to_standard_array_axis_scalar_field(xs)
    ys_aligned = align_to_standard_array_axis_scalar_field(ys)
    scalar_field_aligned = align_to_standard_array_axis_scalar_field(scalar_field)

    contour = ax.contourf(xs_aligned, ys_aligned, scalar_field_aligned, levels=levels)
    return fig, ax, contour

def plot_vector_field_SO3(vector_field, αs, βs, color="red", figsize=(10, 10), fig=None, ax=None, α_min=None,
                          α_max=None, β_min=None, β_max=None):
    """Streamplot of `vector_field`."""
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel("$β$")
        ax.set_ylabel("$α$")
        α_min = overwrite_default(α_min, αs[0, 0])
        α_max = overwrite_default(α_max, αs[-1, -1])
        β_min = overwrite_default(β_min, βs[0, 0])
        β_max = overwrite_default(β_max, βs[-1, -1])
        ax.set_xlim(β_max, β_min)
        ax.set_ylim(α_min, α_max)

    ax.streamplot(βs, αs, vector_field[..., 1], vector_field[..., 0], color=color)
    return fig, ax

def plot_scalar_field_SO3(scalar_field, αs, βs, levels=None, figsize=(12, 10), fig=None, ax=None, α_min=None,
                          α_max=None, β_min=None, β_max=None):
    """Plot two-dimensional `scalar_field` using coloured in contours."""
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel("$β$")
        ax.set_ylabel("$α$")
        α_min = overwrite_default(α_min, αs[0, 0])
        α_max = overwrite_default(α_max, αs[-1, -1])
        β_min = overwrite_default(β_min, βs[0, 0])
        β_max = overwrite_default(β_max, βs[-1, -1])
        ax.set_xlim(β_max, β_min)
        ax.set_ylim(α_min, α_max)

    contour = ax.contourf(βs, αs, scalar_field, levels=levels)
    return fig, ax, contour