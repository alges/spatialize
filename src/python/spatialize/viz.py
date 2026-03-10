"""
Visualization utilities for spatialize results and spatial data.

This module provides:

- **Color palettes** (`PALETTES`): Predefined color dictionaries (hex lists) for
  consistent styling across figures.
- **PlotStyle**: Context-manager class that applies named themes to matplotlib's
  ``rcParams`` and exposes ``color``, ``cmap``, and ``precision_cmap`` attributes.
- **Palette helpers**: ``get_available_palettes``, ``get_palette``,
  ``get_palette_colormap``, ``show_palettes`` — utilities to inspect and create
  colormaps from named palettes or Scientific Colour Maps (SCM).
- **Plot functions**: ``plot_histogram``, ``plot_colormap_data``,
  ``plot_colormap_array``, ``plot_nongriddata``, ``plot_scatter`` — ready-made
  figures for ESI estimation and precision results.

Color palette sources
---------------------
- Scientific colour maps: Crameri, F. (2018). Scientific colour maps. Zenodo.
  https://doi.org/10.5281/zenodo.1243862 — https://www.fabiocrameri.ch/colourmaps/
- Seaborn palettes: Copyright (c) 2012-2023, Michael Waskom. License: BSD-3-Clause.
  https://github.com/mwaskom/seaborn
"""

import numpy as np
import pandas as pd
import random as rd
from spatialize import in_notebook, logging

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap as Colormap
from spatialize.resources import ScientificColourMaps as SCM
from spatialize import SpatializeError
from spatialize.logging import log_message

PALETTES = {
    # ALGES palettes
    'alges': ['#070f15', '#0f212d', '#19384d', '#275474', '#326c94', '#4b87af', '#78abce', '#a7c9e1', '#cfe2ef'],
    'alges_muted': ['#142b3b','#1e4058', '#285676', '#326c94', '#5a89a9', '#84a6be', '#c1d2de'],
    'precision': ['#51648a','#796982','#9d6c79','#c47673','#e98c76','#edb18e','#edd3b3'],
    'precision_dark': ['#0b1425','#133859','#48587a','#6c5e74','#90606c','#bc6461','#e67960','#eb937f','#f2bcaf'],
    'precision_muted': ['#5b6f9a','#867590','#a77c87','#cb8785','#ed9e8c','#f0c0a4','#f2dfc7'],
    # 'crest' palette from seaborn, reversed
    'crest_r': ['#193458', '#254b7f', '#1c6488','#287a8c', '#40908e', '#59a590', '#7dba91','#a4ceb2'],
    }

def _get_scm_colormap(name):
    """
    Get a colormap from the Scientific Colour Maps (SCM) package by name.

    Parameters
    ----------
    name : str
        Name of the SCM palette (e.g., 'batlow', 'navia', 'lipari').
        Append '_r' to get the reversed version.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap or None
        The SCM colormap if found, None otherwise.
    """
    return getattr(SCM, name, None)

def get_available_palettes():
    """
    Get a list of all available palette names, including Scientific Colour Maps.

    Returns
    -------
    list of str
        Names of available color palettes

    Examples
    --------
    >>> palettes = get_available_palettes()
    >>> print(palettes)
    ['alges', 'alges_muted', 'precision', ..., 'batlow', 'navia', ...]
    """
    scm_names = sorted(SCM.__all__) if hasattr(SCM, '__all__') else []
    return list(PALETTES.keys()) + scm_names

def get_palette(name):
    """
    Get a color palette by name.

    Parameters
    ----------
    name : str
        Name of the palette. Use get_available_palettes() to see options.

    Returns
    -------
    list of str
        List of hex color codes in the palette

    Raises
    ------
    ValueError
        If palette name is not found

    Examples
    --------
    >>> colors = get_palette('alges')
    >>> print(colors[0])
    '#070f15'

    >>> # Use palette colors directly
    >>> colors = get_palette('alges_muted')
    >>> plt.scatter(x, y, c=colors[3])
    """
    if name not in PALETTES:
        available = ', '.join(get_available_palettes())
        raise ValueError(f"Palette '{name}' not found. Available palettes: {available}")
    return PALETTES[name]

def get_palette_colormap(name, reverse=False):
    """
    Create a matplotlib colormap from a named palette.

    Parameters
    ----------
    name : str
        Name of the palette. Use get_available_palettes() to see options.
    reverse : bool, optional
        If True, reverses the color order. Default is False.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Matplotlib colormap object ready for use in plotting

    Raises
    ------
    ValueError
        If palette name is not found

    Examples
    --------
    >>> cmap = get_palette_colormap('batlow')
    >>> plt.imshow(data, cmap=cmap)

    >>> # Reverse the colormap
    >>> cmap_r = get_palette_colormap('navia', reverse=True)
    >>> plt.contourf(X, Y, Z, cmap=cmap_r)

    >>> # Use in PlotStyle
    >>> style = PlotStyle(cmap=get_palette_colormap('lipari'))
    >>> plt.scatter(x, y, c=values, cmap=style.cmap)
    """
    if name in PALETTES:
        colors = get_palette(name)
        if reverse:
            colors = colors[::-1]
        return Colormap.from_list(f'{name}{"_r" if reverse else ""}', colors)

    # Try Scientific Colour Maps
    scm_name = f'{name}_r' if reverse else name
    scm_cmap = _get_scm_colormap(scm_name)
    if scm_cmap is not None:
        return scm_cmap

    available = ', '.join(get_available_palettes())
    raise ValueError(f"Palette '{name}' not found. Available palettes: {available}")

def show_palettes(names=None, figsize=None, n_colors=256, title="Available Color Palettes"):
    """
    Display color palettes as horizontal gradient bars for visual comparison.

    Parameters
    ----------
    names : list of str, optional
        Specific palette names to display. If None, shows all available palettes.
    figsize : tuple, optional
        Figure size as (width, height) in inches. If None, auto-calculated based
        on the number of palettes (width=8, height=0.4*n_palettes).
    n_colors : int, optional
        Number of color samples to display for each palette gradient. Default is 256.
    title : str, optional
        Title for the figure. Default is "Available Color Palettes".

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the palette visualizations

    Examples
    --------
    >>> # Show all palettes
    >>> show_palettes()

    >>> # Show specific palettes
    >>> show_palettes(names=['alges', 'precision', 'navia'])

    >>> # Customize figure size
    >>> show_palettes(figsize=(10, 8))

    >>> # Show palettes without title
    >>> show_palettes(title=None)
    """
    # Determine which palettes to show
    if names is None:
        names = get_available_palettes()
    else:
        # Validate provided names
        all_available = set(get_available_palettes())
        invalid = [n for n in names if n not in all_available]
        if invalid:
            raise ValueError(f"Invalid palette names: {invalid}. Use get_available_palettes() to see options.")

    n_palettes = len(names)

    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (8, 0.4 * n_palettes + 0.5)

    # Create figure and axes
    fig, axes = plt.subplots(n_palettes, 1, figsize=figsize)

    # Handle single palette case (axes won't be an array)
    if n_palettes == 1:
        axes = [axes]

    # Create gradient array for displaying colors
    gradient = np.linspace(0, 1, n_colors).reshape(1, -1)

    # Plot each palette
    for ax, name in zip(axes, names):
        # Get colormap and display as gradient
        cmap = get_palette_colormap(name)
        ax.imshow(gradient, aspect='auto', cmap=cmap)

        # Remove ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add palette name as y-label
        ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=10)

    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


class PlotStyle:
    """
    Manage matplotlib plot styles with predefined themes.

    Parameters
    ----------
    theme : str, optional
        Theme name. Available: 'darkgrid', 'whitegrid', 'dark', 'white',
        'alges', 'minimal', 'publication'
    color : str, optional
        Primary color for plots
    cmap : str, list, or colormap, optional
        Colormap for plots. Can be:
        - str: Name from PALETTES (e.g., 'alges'), SCM (e.g., 'batlow'), or matplotlib colormap (e.g., 'viridis')
        - list: List of hex colors to create a custom colormap
        - Colormap object: Matplotlib colormap instance
    precision_cmap : str, list, or colormap, optional
        Colormap for precision/uncertainty plots. Same format as cmap.

    Attributes
    ----------
    theme : str
        Active theme name
    color : str
        Primary plot color
    cmap : str or colormap
        Plot colormap
    precision_cmap : str or colormap
        Precision/uncertainty colormap

    Examples
    --------
    Basic usage with context manager::

        with PlotStyle(theme='dark', color='#ff6b6b') as style:
            plt.plot(x, y, color=style.color)
            plt.imshow(data, cmap=style.cmap)

    Using custom palettes from PALETTES::

        style = PlotStyle(cmap='batlow')  # Uses SCM palette
        plt.imshow(data, cmap=style.cmap)

    Using custom color list::

        custom_colors = ['#ff0000', '#00ff00', '#0000ff']
        style = PlotStyle(cmap=custom_colors)
        plt.scatter(x, y, c=values, cmap=style.cmap)

    Using matplotlib colormaps::

        style = PlotStyle(cmap='viridis')  # Standard matplotlib colormap
        plt.contourf(X, Y, Z, cmap=style.cmap)

    Simple usage without context manager::

        style = PlotStyle(theme='alges')
        plt.plot(x, y, color=style.color)
        style.reset_to_original()       # Manual reset
    """
    THEMES = {
        'darkgrid': {
            'rcparams': {
                'lines.solid_capstyle': 'round',
                'axes.grid': True,
                'axes.facecolor': '#EAEAF2',
                'axes.edgecolor': 'white',
                'axes.labelweight': 'demibold',
                'axes.titleweight': 'bold',
                'figure.titleweight': 'bold',
                'grid.color': 'white',
                'xtick.bottom': False,
                'ytick.left': False,
                'xtick.labelsize': 'small',
                'ytick.labelsize': 'small'
            },
            'color': '#59a590',
            'cmap': get_palette_colormap('crest_r'),
            'precision_cmap': get_palette_colormap('precision')
        },
        'whitegrid': {
            'rcparams': {
                'lines.solid_capstyle': 'round',
                'axes.grid': True,
                'axes.facecolor': 'white',
                'axes.edgecolor': '#EAEAF2',
                'axes.labelweight': 'demibold',
                'axes.titleweight': 'bold',
                'figure.titleweight': 'bold',
                'grid.color': '#EAEAF2',
                'xtick.bottom': False,
                'ytick.left': False,
                'xtick.labelsize': 'small',
                'ytick.labelsize': 'small'
            },
            'color': '#7dba91',
            'cmap': get_palette_colormap('crest_r'),
            'precision_cmap': get_palette_colormap('precision')
        },
        'dark': {
            'rcparams': {
                'figure.facecolor': '#0d121a',
                'axes.facecolor': '#16202c',
                'axes.grid': False,
                'axes.edgecolor': '#343d46',
                'axes.labelcolor': 'white',
                'axes.labelweight': 'demibold',
                'axes.titleweight': 'bold',
                'figure.titleweight': 'bold',
                'text.color': 'white',
                'xtick.color': '#b3b9c1',
                'ytick.color': '#b3b9c1',
                'patch.edgecolor': '#67c9cd',
                'xtick.labelsize': 'small',
                'ytick.labelsize': 'small', 
            },
            'color': '#41bbc0',
            'cmap': SCM.batlowK,
            'precision_cmap': get_palette_colormap('precision_dark')
        },
        'white': {
            'rcparams': {
                'lines.solid_capstyle': 'round',
                'axes.grid': False,
                'axes.facecolor': 'white',
                'axes.edgecolor': '#d2d2d9',
                'axes.titleweight': 'bold',
                'xtick.labelsize': 'small',
                'ytick.labelsize': 'small',
                'axes.labelweight': 'demibold',
                'figure.titleweight': 'bold',
                'xtick.bottom': True,
                'ytick.left': True,
            },
            'color': '#7dba91',
            'cmap': get_palette_colormap('crest_r'),
            'precision_cmap': get_palette_colormap('precision')
        },
        'alges': {
            'rcparams': {
                'lines.solid_capstyle': 'round',
                'axes.grid': False,
                'axes.facecolor': 'white',
                'axes.edgecolor': '#808080',
                'axes.labelcolor': '#424e77',
                'axes.labelweight': 'demibold',
                'axes.titleweight': 'bold',
                'figure.titleweight': 'bold',
                'xtick.labelsize': 'small',
                'ytick.labelsize': 'small',
                'xtick.color': '#808080',
                'ytick.color': '#808080',
                'patch.edgecolor': '#496070',
                'text.color': "#424e77",
                'xtick.bottom': True,
                'ytick.left': True
            },
            'color': "#8fb4cd",
            'cmap': get_palette_colormap('alges'),
            'precision_cmap': get_palette_colormap('precision')
        },
        'alges_muted': {
            'rcparams': {
                'lines.solid_capstyle': 'round',
                'axes.grid': False,
                'axes.facecolor': 'white',
                'axes.edgecolor': '#808080',
                'axes.labelcolor': '#545f84',
                'axes.labelweight': 'demibold',
                'axes.titleweight': 'bold',
                'figure.titleweight': 'bold',
                'xtick.labelsize': 'small',
                'ytick.labelsize': 'small',
                'xtick.color': '#808080',
                'ytick.color': '#808080',
                'patch.edgecolor': '#496070',
                'text.color': "#424e77",
                'xtick.bottom': True,
                'ytick.left': True
            },
            'color': "#84a6be",
            'cmap': get_palette_colormap('alges_muted'),
            'precision_cmap': get_palette_colormap('precision_muted')
        },
        'minimal': {
            'rcparams': {
                'axes.grid': False,
                'axes.spines.top': False,
                'axes.spines.right': False,
            },
            'color': '#333333',
            'cmap': 'binary_r',
            'precision_cmap': 'binary_r',
        },
        'publication': {
            'rcparams': {
                'figure.titleweight': 'bold',
                'axes.grid': True,
                'grid.color': 'white',
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.titleweight': 'bold',
                'font.family': 'serif',
                'font.size': 10,
            },
            'color': 'black',
            'cmap': 'viridis',
            'precision_cmap': 'cividis'
        }
    }

    DEFAULT_COLOR = 'skyblue'
    DEFAULT_CMAP = 'coolwarm'
    DEFAULT_PRECISION_CMAP = 'bwr'

    def __init__(self,
                 theme = None,
                 color = None,
                 cmap = None,
                 precision_cmap = None):
        
        if theme and theme not in self.THEMES:
            raise ValueError(f"Theme '{theme}' not found. Available: {list(self.THEMES.keys())}")
        
        self._original_rcparams = plt.rcParams.copy()

        self.theme = theme
        self.color = self._set_color(color)
        self.cmap = self._set_cmap(cmap)
        self.precision_cmap = self._set_precision_cmap(precision_cmap)

        if self.theme:
            self._apply_theme()

    def _set_color(self, color):
        if color:
            return color
        elif self.theme:
            return self.THEMES[self.theme]['color']
        else:
            return self.DEFAULT_COLOR
    
    def _set_cmap(self, cmap):
        if cmap:
            # Handle string input - check PALETTES first, then SCM, then matplotlib
            if isinstance(cmap, str):
                if cmap in PALETTES:
                    return get_palette_colormap(cmap)
                scm_cmap = _get_scm_colormap(cmap)
                if scm_cmap is not None:
                    return scm_cmap
                # Return as-is, assuming it's a matplotlib colormap name
                return cmap
            # Handle list input - create colormap from color list
            elif isinstance(cmap, list):
                return Colormap.from_list('custom_cmap', cmap)
            # Otherwise return as-is (already a colormap object)
            else:
                return cmap
        elif self.theme:
            return self.THEMES[self.theme]['cmap']
        else:
            return self.DEFAULT_CMAP

    def _set_precision_cmap(self, cmap):
        if cmap:
            # Handle string input - check PALETTES first, then SCM, then matplotlib
            if isinstance(cmap, str):
                if cmap in PALETTES:
                    return get_palette_colormap(cmap)
                scm_cmap = _get_scm_colormap(cmap)
                if scm_cmap is not None:
                    return scm_cmap
                # Return as-is, assuming it's a matplotlib colormap name
                return cmap
            # Handle list input - create colormap from color list
            elif isinstance(cmap, list):
                return Colormap.from_list('custom_precision_cmap', cmap)
            # Otherwise return as-is (already a colormap object)
            else:
                return cmap
        elif self.theme:
            return self.THEMES[self.theme]['precision_cmap']
        else:
            return self.DEFAULT_PRECISION_CMAP

    def _apply_theme(self):
        if self.theme and self.theme in self.THEMES:
            theme_config = self.THEMES[self.theme]['rcparams']
            plt.rcParams.update(theme_config)
    
    def get_available_themes(self):
        """
        Return the names of all built-in themes.

        Returns
        -------
        list of str
            Available theme names (e.g. ``['darkgrid', 'alges', 'publication', ...]``).
        """
        return list(self.THEMES.keys())

    def reset_to_original(self) -> None:
        """
        Restore matplotlib ``rcParams`` to the state captured at construction.

        This is called automatically when exiting the context manager, but can
        also be invoked manually when :class:`PlotStyle` is used without
        ``with``.
        """
        plt.rcParams.update(self._original_rcparams)

    def get_theme_info(self, theme_name: str):
        """
        Return the configuration dict for a named theme.

        Parameters
        ----------
        theme_name : str
            Name of the theme to inspect.

        Returns
        -------
        dict
            A copy of the theme configuration containing keys
            ``'rcparams'``, ``'color'``, ``'cmap'``, and
            ``'precision_cmap'``.

        Raises
        ------
        ValueError
            If *theme_name* is not a recognised theme.
        """
        if theme_name not in self.THEMES:
            raise ValueError(f"Theme '{theme_name}' not found.")
        return self.THEMES[theme_name].copy()

    def __repr__(self):
        return f"PlotStyle(theme='{self.theme}', color='{self.color}', cmap='{self.cmap}')"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset_to_original()

def _is_xy_indexed(xi):
    """
    Returns True if xi is a grid tuple using np.meshgrid xy-indexing
    (x varies along axis 1). Returns False for np.mgrid ij-indexing
    (x varies along axis 0) or for non-grid inputs.

    Detection: in ij-indexing the first row of grid_x is constant;
    in xy-indexing it varies.
    """
    if xi is None or not isinstance(xi, (tuple, list)) or len(xi) < 2:
        return False
    gx = np.asarray(xi[0])
    if gx.ndim < 2 or gx.shape[1] < 2:
        return False
    return not np.allclose(gx[0, :], gx[0, 0])


def plot_histogram(data, ax, color='skyblue', alpha=0.9, rwidth=0.92, hide_empty=True):
    """
    Plot a histogram of numerical data onto an existing axes.

    Parameters
    ----------
    data : array-like, shape (n,)
        1D array of numerical values (list, numpy array, or pandas Series).
    ax : matplotlib.axes.Axes
        Axes object on which to draw the histogram.
    color : str, optional
        Bar fill color. Default is ``'skyblue'``.
    alpha : float, optional
        Bar transparency in [0, 1]. Default is ``0.9``.
    rwidth : float, optional
        Relative bar width in (0, 1]. Default is ``0.92``.
    hide_empty : bool, optional
        If ``True``, tick labels for empty bins are hidden. Default is ``True``.

    Returns
    -------
    None
        The histogram is drawn in-place on *ax*.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> plot_histogram(errors, ax, color='steelblue')
    """
    counts, bin_edges, _ = ax.hist(data,
                                   color=color,
                                   alpha=alpha,
                                   rwidth=rwidth,
                                   zorder=3)
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    if hide_empty:
        ax.set_xticks(bin_centers[counts > 0])
    else:
        ax.set_xticks(bin_centers)
    ax.tick_params(axis='x', rotation=30)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))

def plot_colormap_data(data, ax=None, w=None, h=None, xi_locations=None, griddata=False,
                       title=None, xlabel='', ylabel='', show_colorbar=True, cbar_label='Value',
                       figsize=None, dpi=120, **imshow_args):
    """
    Plot a colormap (heatmap) visualization of spatial data.

    Handles three data layouts:

    - **Grid data** (``griddata=True``): *data* is already a 2D array; it is
      transposed for display.
    - **Explicit shape** (``w`` and ``h`` provided): flat *data* is reshaped to
      ``(h, w)``.
    - **Inferred shape** (``xi_locations`` provided): shape is derived from the
      unique X/Y coordinates in *xi_locations*.

    Parameters
    ----------
    data : array-like
        Values to visualize. Either a 2D grid array (when ``griddata=True``) or
        a flat 1D array that can be reshaped to ``(h, w)``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If ``None``, a new figure and axes are created.
    w : int, optional
        Number of columns (image width). Required when ``griddata=False`` and
        ``xi_locations`` is not provided.
    h : int, optional
        Number of rows (image height). Required when ``griddata=False`` and
        ``xi_locations`` is not provided.
    xi_locations : array-like, shape (n, 2), optional
        Spatial coordinates used to infer ``(h, w)`` when neither is given
        explicitly.
    griddata : bool, optional
        If ``True``, *data* is treated as a 2D grid and transposed before
        display. Default is ``False``.
    title : str, optional
        Figure title (only used when *ax* is ``None``). Default is ``""``.
    show_colorbar : bool, optional
        Whether to add a colorbar. Default is ``True``.
    cbar_label : str, optional
        Label for the colorbar. Default is ``'Value'``.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches (only used when *ax* is
        ``None``).
    dpi : int, optional
        Figure resolution in dots-per-inch (only used when *ax* is ``None``).
        Default is ``120``.
    **imshow_args
        Additional keyword arguments forwarded to :func:`matplotlib.pyplot.imshow`,
        e.g. ``cmap``, ``vmin``, ``vmax``.

    Returns
    -------
    None
        The plot is drawn in-place on the axes.

    Raises
    ------
    SpatializeError
        If ``griddata=False`` and neither ``w``/``h`` nor ``xi_locations`` are
        provided, or if the data cannot be reshaped to the inferred dimensions.

    Examples
    --------
    >>> # Grid data (output of esi_griddata)
    >>> plot_colormap_data(result.estimation(), griddata=True, cmap='viridis')

    >>> # Scattered points with explicit grid dimensions
    >>> plot_colormap_data(values, w=50, h=50, cmap='coolwarm')

    >>> # Scattered points with inferred dimensions
    >>> plot_colormap_data(values, xi_locations=xi, cmap='batlow',
    ...                   title='ESI Estimation')
    """
    if griddata:
        im = data if _is_xy_indexed(xi_locations) else data.T
    elif w is not None and h is not None:
        im = data.reshape(h, w)
    elif xi_locations is None:
        raise SpatializeError("Must provide either w/h or xi_locations")
    else:
        # Ensure data is in correct order for imshow
        sort_indices = np.lexsort((xi_locations[:, 0], xi_locations[:, 1]))
        sorted_data = data[sort_indices]
        
        w, h = len(np.unique(xi_locations[:, 0])), len(np.unique(xi_locations[:, 1]))
        if len(data) != w * h:
            w, h = w-1, h-1
        log_message(logging.logger.debug(f"using h={h}, w={w}"))
            
        im = sorted_data.reshape(h, w)

    if ax is not None:
        plotter = ax
    else:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(1, 1)
        plotter = gs.subplots()
    _ax_format(plotter, xlabel, ylabel, title)

    img = plotter.imshow(im, origin='lower', **imshow_args)

    # Optional colorbar
    if show_colorbar:
        divider = make_axes_locatable(plotter)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cax.grid(False)
        cbar = plt.colorbar(img, orientation='vertical', cax=cax)
        cbar.set_label(cbar_label)
    
def plot_colormap_array(data, n_imgs=9, n_cols=3, norm_lims=False, xi_locations=None,
                        reference_map=None, title=None, title_prefix="scenario", seed=None,
                        figsize=(10, 10), dpi=120, **imshow_args):
    """
    Plot a grid of colormap visualizations for a random subset of data columns.

    Useful for inspecting multiple ESS simulation realisations or ESI ensemble
    members side-by-side. A random subset of ``n_imgs`` columns is sampled from
    *data* and displayed in a ``(n_rows × n_cols)`` subplot grid.

    Parameters
    ----------
    data : array-like, shape (n_points, n_scenarios)
        2D array where each column is a separate spatial map to display.
    n_imgs : int, optional
        Number of subplots to draw. Clamped to ``data.shape[1]`` if larger.
        Default is ``9``.
    n_cols : int, optional
        Number of columns in the subplot grid. Default is ``3``.
    norm_lims : bool, optional
        If ``True`` and *reference_map* is provided, sets shared ``vmin``/``vmax``
        from the reference map to make panels visually comparable. Default is
        ``False``.
    xi_locations : array-like, shape (n_points, 2), optional
        Spatial coordinates used to reshape each column for display. Passed
        directly to :func:`plot_colormap_data`.
    reference_map : array-like, optional
        Map used to derive shared colormap limits when ``norm_lims=True``.
        ``vmin = nanmin + 0.1`` and ``vmax = nanmax - 0.1``.
    title : str, optional
        Super-title for the entire figure. If ``None``, no title is added.
    title_prefix : str, optional
        Prefix for individual subplot titles, e.g. ``"scenario"`` produces
        titles like *"scenario 3"*. Default is ``"scenario"``.
    seed : int, optional
        Random seed for column sampling. If ``None``, a random seed is generated
        and returned so results can be reproduced.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches. Default is ``(10, 10)``.
    dpi : int, optional
        Figure resolution in dots-per-inch. Default is ``120``.
    **imshow_args
        Additional keyword arguments forwarded to :func:`plot_colormap_data`
        (and thus to :func:`matplotlib.pyplot.imshow`), e.g. ``cmap``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object (only when not running inside a Jupyter notebook).
    seed : int
        The random seed used for column sampling, allowing reproduction of the
        same panel layout.

    Notes
    -----
    When called inside a Jupyter notebook (:func:`in_notebook` returns
    ``True``), only *seed* is returned (the figure is displayed inline).

    Examples
    --------
    >>> fig, seed = plot_colormap_array(sim_results, n_imgs=6, n_cols=3,
    ...                                 xi_locations=xi, cmap='batlow',
    ...                                 title='ESS Simulations')
    >>> print(f"Reproduced with seed={seed}")
    """
    if n_imgs > data.shape[1]:
        n_imgs = data.shape[1]

    n_rows = n_imgs // n_cols if n_imgs % n_cols == 0 else (n_imgs // n_cols + 1)

    if seed is not None:
        rd.seed(seed)
    else:
        # if no seed is provided, we generate and capture a random one
        # to make the results reproducible
        rd.seed()
        seed = rd.randint(0, 10000)
        rd.seed(seed)

    # random sample of colormap images to plot
    sim_idx = rd.sample(range(data.shape[1]), k=n_imgs)

    # set kwargs for the plot
    plot_imshow_args = imshow_args.copy()
    plot_imshow_args.setdefault('cmap', 'coolwarm')

    if norm_lims and reference_map is not None:
        # if a reference map is provided, we use its min and max values to make the images
        # more comparable and we normalize the values, sharpening the range of the data
        plot_imshow_args['vmin'] = np.nanmin(reference_map) + 0.1
        plot_imshow_args['vmax'] = np.nanmax(reference_map) - 0.1
    else:
        plot_imshow_args.setdefault('vmin', None)
        plot_imshow_args.setdefault('vmax', None)

    # plot the simulation results
    fig = plt.figure(dpi=dpi, figsize=figsize)
    gs = fig.add_gridspec(n_rows, n_cols)
    subplots_result = gs.subplots()
    axes = (np.asarray(subplots_result).flatten()
            if isinstance(subplots_result, np.ndarray)
            else np.array([subplots_result]))

    for i in range(n_imgs):
        plot_colormap_data(data[:, sim_idx[i]], ax=axes[i],
                        xi_locations=xi_locations,
                        **plot_imshow_args)
        axes[i].set_title(f"{title_prefix} {sim_idx[i] + 1}")

    for i in range(n_imgs, n_rows * n_cols):
        axes[i].set_axis_off()
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    if title:
        fig.suptitle(title, fontsize=14)

    if in_notebook():
        return seed

    return fig, seed


def plot_nongriddata(data, xi_locations, ax=None, title="",
                       figsize=None, dpi=120, origin='lower', **imshow_args):
    """
    Plots a colormap visualization for non-rectangular/irregular spatial data.

    This function handles spatial data that doesn't form a complete rectangular grid,
    such as irregularly shaped study areas (e.g., lakes, watersheds). It uses xarray
    to organize scattered points into a plottable format, automatically handling gaps
    in the spatial domain.

    Parameters
    ----------
    data : array-like, shape (n_points,)
        1D array of values to visualize at each spatial location.
    xi_locations : array-like, shape (n_points, 2)
        2D array of spatial coordinates (X, Y) for each data point.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot into. If None, creates a new figure and axes.
    title : str, optional
        Title to display on the plot (only used when ax is None). Default is "".
    figsize : tuple, optional
        Width, height of the figure in inches (only used when ax is None).
    dpi : int, optional
        Resolution of the figure in dots-per-inch (only used when ax is None).
        Default is 120.
    origin : {'lower', 'upper'}, optional
        Controls the orientation of the Y-axis:
        - 'lower': Y increases upward (standard Cartesian, default)
        - 'upper': Y increases downward (image/raster convention)
    **imshow_args : dict
        Additional keyword arguments passed to imshow(), such as 'cmap', 'vmin', 'vmax'.

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the figure object if ax is None and not in notebook mode,
        otherwise returns None.

    Raises
    ------
    SpatializeError
        If xi_locations is not 2-dimensional or doesn't have exactly 2 columns.
    SpatializeError
        If data and xi_locations have incompatible shapes.

    Examples
    --------
    >>> # Plot irregular spatial data (standard Cartesian orientation)
    >>> xi_locations = np.array([[0, 0], [1, 0], [0, 1], [2, 1]])  # Irregular points
    >>> data = np.array([1.0, 2.0, 1.5, 3.0])
    >>> plot_nongriddata(data, xi_locations, title="Lake Temperature")

    >>> # Use image/raster convention (Y increases downward)
    >>> plot_nongriddata(data, xi_locations, origin='upper', title="Raster Data")

    >>> # Use custom colormap and value range
    >>> fig, ax = plt.subplots()
    >>> plot_nongriddata(data, xi_locations, ax=ax, cmap='viridis', vmin=0, vmax=5)
    """
    # Validate xi_locations dimensions
    if xi_locations.ndim != 2:
        raise SpatializeError(
            f"xi_locations must be a 2D array, got {xi_locations.ndim}D array"
        )

    if xi_locations.shape[1] != 2:
        raise SpatializeError(
            f"xi_locations must have exactly 2 columns (X, Y), got {xi_locations.shape[1]}"
        )

    # Validate data compatibility
    data = np.asarray(data).flatten()
    if len(data) != len(xi_locations):
        raise SpatializeError(
            f"data length ({len(data)}) must match xi_locations length ({len(xi_locations)})"
        )

    # Create DataFrame with spatial coordinates and values
    results = pd.DataFrame(xi_locations, columns=['X', 'Y'])
    results['est'] = data

    # Convert to xarray using X, Y as index dimensions
    # This automatically handles irregular/sparse grids
    results_array = results.set_index(['X', 'Y']).to_xarray()

    # Create or use provided axes
    fig_created = False
    if ax is not None:
        plotter = ax
    else:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(1, 1)
        plotter = gs.subplots()
        plotter.set_title(title)
        fig_created = True

    # Plot the data
    # xarray creates array with dimensions (X, Y), but imshow expects (Y, X)
    # Transpose to swap dimensions; origin parameter controls Y-axis direction
    img = plotter.imshow(results_array.est.T, origin=origin, **imshow_args)

    # Add colorbar
    divider = make_axes_locatable(plotter)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cax.grid(False)
    plt.colorbar(img, orientation='vertical', cax=cax)

    # Return figure if we created it and not in notebook
    if fig_created and not in_notebook():
        return fig
    

def plot_scatter(points, values, cmap=None,
                 xlabel='X', ylabel='Y', title=None,
                 theme='publication',
                 figsize=None, dpi=100,
                 fig=None, ax=None,
                 show_colorbar=True, cbar_label='Value',
                 **plot_args):
    """
    Create a 2D scatter plot of spatial data coloured by value.

    Applies a :class:`PlotStyle` theme and renders a scatter plot where point
    colour encodes *values*. An optional colorbar is appended to the right of
    the axes.

    Parameters
    ----------
    points : array-like, shape (n, 2)
        Spatial coordinates. Column 0 → X axis, column 1 → Y axis.
    values : array-like, shape (n,)
        Scalar values used to colour each point.
    cmap : str, list, or colormap, optional
        Colormap override. Accepts the same formats as :class:`PlotStyle`
        ``cmap`` parameter (palette name, hex list, or matplotlib colormap).
        If ``None``, the theme default is used.
    xlabel : str, optional
        X-axis label. Default is ``'X'``.
    ylabel : str, optional
        Y-axis label. Default is ``'Y'``.
    title : str, optional
        Axes title. Default is ``None``.
    theme : str, optional
        :class:`PlotStyle` theme name. Default is ``'publication'``.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches.
    dpi : int, optional
        Figure resolution in dots-per-inch. Default is ``100``.
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into. Used together with *ax*.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into. If ``None``, a new figure and axes are
        created.
    show_colorbar : bool, optional
        Whether to append a colorbar. Default is ``True``.
    cbar_label : str, optional
        Colorbar label. Default is ``'Value'``.
    **plot_args
        Additional keyword arguments forwarded to
        :func:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the scatter plot.
    ax : matplotlib.axes.Axes
        The axes containing the scatter plot.

    Examples
    --------
    >>> fig, ax = plot_scatter(points, values, cmap='batlow',
    ...                        title='ESI Estimation', theme='alges')
    >>> plt.show()

    >>> # Overlay on existing axes
    >>> fig, ax = plt.subplots()
    >>> plot_scatter(points, values, ax=ax, fig=fig, show_colorbar=False)
    """

    X = points[:, 0]
    Y = points[:, 1]
    extent = [np.min(X),  np.max(X),  np.min(Y),  np.max(Y)]

    with PlotStyle(theme=theme, cmap=cmap) as style:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        plot = ax.scatter(X, Y, c=values, cmap=style.cmap, zorder=3, **plot_args)
        _ax_format(ax, xlabel, ylabel, title, extent)

        # Optional colorbar
        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cax.grid(False)
            cbar = plt.colorbar(plot, orientation='vertical', cax=cax)
            cbar.set_label(cbar_label)

    return fig, ax

def _ax_format(ax, xlabel, ylabel, title, extent=None):
    """
    Apply standard formatting to a spatial axes.

    Sets axis labels, title, and optionally axis limits and equal aspect ratio.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    title : str or None
        Axes title (``loc='center'``).
    extent : sequence of float, length 4, optional
        ``[x_min, x_max, y_min, y_max]`` used for :meth:`set_xlim`,
        :meth:`set_ylim`, and :meth:`set_aspect`. When ``None`` (default),
        limits and aspect are left unchanged (useful for image/imshow axes).
    """
    if title:
        ax.set_title(title, loc='center')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if extent is not None:
        x_min, x_max, y_min, y_max = extent
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')


# --- categorical ---
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def _sort_categories(categories, custom_order=None):
    """Sort categories numerically or alphabetically (strings last)."""
    order_map = {item: i for i, item in enumerate(custom_order)} if custom_order else None

    def key(item):
        try:
            return (0, float(item))
        except (ValueError, TypeError):
            s = str(item)
            if order_map is not None:
                return (1, order_map.get(s, float('inf')))
            return (1, s)

    return sorted(list(categories), key=key)


def _resolve_categorical_cmap(cmap, n):
    """
    Resolve a colormap for categorical data and return *n* hex colour strings.

    Accepts the same *cmap* types supported by ``PlotStyle`` and
    ``plot_colormap_data``:

    - ``None`` / ``'auto'``   – automatic fallback: Accent (≤10), tab20 (≤20),
                                plasma (≤40), turbo (>40).
    - ``str``                 – name of a spatialize PALETTE, an SCM colormap,
                                or any matplotlib colormap.
    - matplotlib Colormap     – used directly (sampled at *n* evenly-spaced points).
    - list of str (hex/names) – used as-is, cycling if ``len(list) < n``.
    """
    import matplotlib.colors as mcolors

    # --- list of explicit colours ---
    if isinstance(cmap, list):
        colors = [cmap[i % len(cmap)] for i in range(n)]
        return colors

    # --- matplotlib Colormap object ---
    if isinstance(cmap, mcolors.Colormap):
        cm = cmap
        return ['#%02x%02x%02x' % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
                for c in (cm(t) for t in np.linspace(0, 1, n))]

    # --- None / 'auto' → legacy automatic fallback ---
    if cmap is None or cmap == 'auto':
        if n <= 10:
            cm = plt.colormaps['Accent']
        elif n <= 20:
            cm = plt.colormaps['tab20']
        elif n <= 40:
            cm = plt.colormaps['plasma']
        else:
            cm = plt.colormaps['turbo']
        return ['#%02x%02x%02x' % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
                for c in (cm(t) for t in np.linspace(0, 1, n))]

    # --- string name: PALETTES → SCM → matplotlib ---
    if isinstance(cmap, str):
        if cmap in PALETTES:
            cm = get_palette_colormap(cmap)
        else:
            scm_cmap = _get_scm_colormap(cmap)
            if scm_cmap is not None:
                cm = scm_cmap
            else:
                cm = plt.colormaps.get_cmap(cmap)   # raises ValueError for unknown names
        return ['#%02x%02x%02x' % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
                for c in (cm(t) for t in np.linspace(0, 1, n))]

    raise ValueError(
        f"Unsupported cmap type {type(cmap).__name__!r}. "
        "Pass a string (palette/SCM/matplotlib name), a matplotlib Colormap, "
        "a list of colours, or None for automatic selection."
    )


def _category_colors(n, cmap='Accent'):
    return _resolve_categorical_cmap(cmap, n)


# ─────────────────────────────────────────────────────────────
#  Categorical scatter plot
# ─────────────────────────────────────────────────────────────

def plot_categorical_scatter(points, values, cmap='Accent',
                              nonnum_order=None,
                              xlabel='X', ylabel='Y', title=None,
                              theme='publication',
                              figsize=None, dpi=100,
                              fig=None, ax=None,
                              extent=None,
                              legend_title='Category',
                              **scatter_args):
    """
    2D scatter plot coloured by categorical labels.

    Based on ``spatialize.viz.plot_scatter`` but uses a per-category discrete
    colour scheme with a legend instead of a continuous colourbar.

    Parameters
    ----------
    points : (N, 2) array of 2D coordinates
    values : (N,) categorical labels
    cmap : str | matplotlib Colormap | list of colours | None
        Colour source for the discrete category palette.  Accepts:

        - ``None`` (default ``'Accent'`` for ≤10 categories, with automatic
          fallback to ``tab20`` / ``plasma`` / ``turbo`` for larger sets).
        - A **spatialize palette** name (e.g. ``'alges'``, ``'crest_r'``).
        - A **Scientific Colour Map** name (e.g. ``'batlow'``, ``'roma'``).
        - Any **matplotlib** colourmap name (e.g. ``'viridis'``, ``'tab20'``).
        - A matplotlib **Colormap object** (sampled at *n* evenly-spaced points).
        - A **list of hex/named colours** cycled to cover all categories.
    nonnum_order : list, optional custom sort order for non-numeric categories
    xlabel, ylabel, title : str, axis labels and title
    theme : str, PlotStyle theme name (default 'publication')
    figsize, dpi : figure size and resolution
    fig, ax : existing matplotlib Figure/Axes to plot into
    legend_title : str, title for the legend
    **scatter_args : forwarded to ``ax.scatter()``

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise SpatializeError("plot_categorical_scatter only supports 2D data.")

    unique_cats = _sort_categories(np.unique(values), nonnum_order)
    colors = _category_colors(len(unique_cats), cmap)

    X = points[:, 0]
    Y = points[:, 1]
    if not extent:
        extent = [X.min(), X.max(), Y.min(), Y.max()]

    with PlotStyle(theme=theme):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for i, cat in enumerate(unique_cats):
            mask = values == cat
            ax.scatter(X[mask], Y[mask], label=str(cat), color=colors[i],
                       zorder=3, **scatter_args)

        _ax_format(ax, xlabel, ylabel, title, extent)
        ax.legend(title=legend_title, loc='center left', bbox_to_anchor=(1.05, 0.5))

    return fig, ax


# ─────────────────────────────────────────────────────────────
#  Categorical colormap (image) plot
# ─────────────────────────────────────────────────────────────

def plot_categorical_colormap(data, categories=None, cmap='Accent',
                               ax=None, w=None, h=None, xi_locations=None,
                               griddata=False, nonnum_order=None,
                               extent=None, title=None, legend_title='Category',
                               figsize=None, dpi=120, **kwargs):
    """
    Image-style categorical map using a discrete ``ListedColormap``.

    Based on ``spatialize.viz.plot_colormap_data`` but for categorical labels:
    categories are mapped to integer codes, displayed with a discrete colourmap,
    and a legend replaces the continuous colourbar.

    Parameters
    ----------
    data : array-like
        Categorical labels. Either 2D (shape h×w) if ``griddata=True``, or 1D flat.
    categories : list, optional
        Ordered list of unique categories. Inferred from *data* if None.
    cmap : str | matplotlib Colormap | list of colours | None
        Colour source for the discrete category palette.  Accepts:

        - ``None`` (default ``'Accent'`` for ≤10 categories, with automatic
          fallback to ``tab20`` / ``plasma`` / ``turbo`` for larger sets).
        - A **spatialize palette** name (e.g. ``'alges'``, ``'crest_r'``).
        - A **Scientific Colour Map** name (e.g. ``'batlow'``, ``'roma'``).
        - Any **matplotlib** colourmap name (e.g. ``'viridis'``, ``'tab20'``).
        - A matplotlib **Colormap object** (sampled at *n* evenly-spaced points).
        - A **list of hex/named colours** cycled to cover all categories.
    ax : matplotlib Axes, optional. Created if None.
    w, h : int, grid width/height for reshaping (1D input only).
    xi_locations : (N, 2) array, used to infer grid shape when ``griddata=False``
        and ``w``/``h`` are not given — same semantics as ``plot_colormap_data``.
    griddata : bool
        If True, *data* is already 2D (h×w) and will be transposed for display.
    nonnum_order : list, optional custom sort order for non-numeric categories.
    extent : list [x_min, x_max, y_min, y_max], optional, passed to imshow.
    title : str, plot title (only when ``ax`` is None).
    legend_title : str, legend title.
    figsize, dpi : figure size / resolution (only when ``ax`` is None).
    """
    data = np.asarray(data, dtype=object)
    flat_data = data.ravel()

    # Determine ordered categories from data if not provided
    if categories is None:
        valid = [v for v in flat_data
                 if v is not None and not (isinstance(v, float) and np.isnan(v))]
        unique_originals = list(dict.fromkeys(valid))   # preserve first-seen, dedup
        categories = _sort_categories(unique_originals, nonnum_order)

    n_cats = len(categories)
    colors = _category_colors(n_cats, cmap)
    cat_to_code = {str(cat): i for i, cat in enumerate(categories)}

    # Encode: category label → int code; unknown/None/NaN → NaN
    encoded = np.array(
        [cat_to_code.get(str(v), np.nan)
         if v is not None and not (isinstance(v, float) and np.isnan(v))
         else np.nan
         for v in flat_data],
        dtype=float,
    )

    # Reshape to 2D image (same logic as plot_colormap_data)
    if griddata:
        reshaped = encoded.reshape(data.shape)
        im = reshaped if _is_xy_indexed(xi_locations) else reshaped.T
    elif w is not None and h is not None:
        im = encoded.reshape(h, w)
    elif xi_locations is not None:
        sort_indices = np.lexsort((xi_locations[:, 0], xi_locations[:, 1]))
        sorted_enc = encoded[sort_indices]
        w_u = len(np.unique(xi_locations[:, 0]))
        h_u = len(np.unique(xi_locations[:, 1]))
        if len(encoded) != w_u * h_u:
            w_u, h_u = w_u - 1, h_u - 1
        log_message(logging.logger.debug(f"plot_categorical_colormap: h={h_u}, w={w_u}"))
        im = sorted_enc.reshape(h_u, w_u)
    else:
        raise SpatializeError(
            "Must provide either w/h or xi_locations when griddata=False."
        )

    # Create axes if needed
    if ax is not None:
        plotter = ax
    else:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(1, 1)
        plotter = gs.subplots()
    if title:
        plotter.set_title(title)

    # Discrete colormap — each integer code maps to one colour
    listed_cmap = ListedColormap(colors[:n_cats])
    imshow_kwargs = dict(origin='lower', cmap=listed_cmap,
                         vmin=-0.5, vmax=n_cats - 0.5,
                         interpolation='none')
    if extent is not None:
        imshow_kwargs['extent'] = extent
    # User kwargs are accepted but categorical display settings take precedence
    plotter.imshow(im, **{**kwargs, **imshow_kwargs})

    # Discrete legend instead of a continuous colourbar
    legend_elements = [
        Patch(facecolor=colors[i], label=str(cat))
        for i, cat in enumerate(categories)
    ]
    plotter.legend(handles=legend_elements, title=legend_title,
                   loc='center left', bbox_to_anchor=(1.02, 0.5))
