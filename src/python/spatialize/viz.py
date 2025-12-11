import numpy as np
import random as rd
from spatialize import in_notebook, logging

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap as Colormap

from spatialize import SpatializeError
from spatialize.logging import log_message

"""
Color palettes for data visualization.

Some of the following palettes are derived from:
- Scientific colour maps: Crameri, F. (2018). Scientific colour maps. Zenodo. https://doi.org/10.5281/zenodo.1243862
  Source: https://www.fabiocrameri.ch/colourmaps/

- Seaborn palettes: Copyright (c) 2012-2023, Michael Waskom
  License: BSD-3-Clause
  Source: https://github.com/mwaskom/seaborn
"""

PALETTES = {
    # ALGES palettes
    'alges': ['#070f15', '#0f212d', '#19384d', '#275474', '#326c94', '#4b87af', '#78abce', '#a7c9e1', '#cfe2ef'],
    'alges_muted': ['#142b3b','#1e4058', '#285676', '#326c94', '#5a89a9', '#84a6be', '#c1d2de'],
    'precision': ['#51648a','#796982','#9d6c79','#c47673','#e98c76','#edb18e','#edd3b3'],
    'precision_dark': ['#0b1425','#133859','#48587a','#6c5e74','#90606c','#bc6461','#e67960','#eb937f','#f2bcaf'],
    'precision_muted': ['#5b6f9a','#867590','#a77c87','#cb8785','#ed9e8c','#f0c0a4','#f2dfc7'],
    # 'crest' palette from seaborn, reversed
    'crest_r': ['#193458', '#254b7f', '#1c6488','#287a8c', '#40908e', '#59a590', '#7dba91','#a4ceb2'],
    # Scientific colour maps
    'navia': ['#0b1627','#16345c','#19598c','#29738e','#398285','#4b9379','#66aa6a','#98ca6e','#d9e5a6','#fcf5d9'],
    'batlow': ['#011a5a', '#104261', '#205f61', '#4c734d', '#828434', '#c09036', '#f2a069', '#fcb3b3', '#fbcdfa'],
    'batlowK': ['#011a5a','#104261','#205f61','#4c734d','#828434','#c09036','#f2a069','#fcb3b3','#fee5e5'],
    'glasgow': ['#371437','#4e1921','#6a2810','#74471c','#716328','#697c47','#61917c','#74a8af','#a8bed7','#dcd1e8'],
    'lipari': ['#0b1425','#133859','#48587a','#6c5e74','#90606c','#bc6461','#e67960','#e9a278','#e8c79e','#fef5db'],
    'nuuk': ['#11598c','#2c6384','#4b7182','#70868c','#939b96','#acad95','#bbb98b','#c7c581','#e1e08c','#fbf7b3'],
    'bamako': ['#073a46','#12433f','#214f33','#385d2c','#547032','#738437','#978e33','#bfa830','#e2c66b','#fee4ab'],
    'tokio': ['#1f1032','#4b1f42','#6a404e','#715651','#746651','#767a54','#7d9857','#8ec26d','#c3e0a7','#eff5db'],
    'bilbao': ['#471010' ,'#752329', '#94454b', '#a16157', '#a6775a', '#ac8c5f', '#b6a672', '#c2bca1', '#d4d1cd', '#ffffff']
    }

def get_available_palettes():
    """
    Get a list of all available palette names.

    Returns
    -------
    list of str
        Names of available color palettes

    Examples
    --------
    >>> palettes = get_available_palettes()
    >>> print(palettes)
    ['alges', 'alges_muted', 'precision', ...]
    """
    return list(PALETTES.keys())

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
    >>> colors = get_palette('glasgow')
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
    colors = get_palette(name)
    if reverse:
        colors = colors[::-1]
    return Colormap.from_list(f'{name}{"_r" if reverse else ""}', colors)

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
        invalid = [n for n in names if n not in PALETTES]
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

    if not in_notebook():
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
        - str: Name from PALETTES (e.g., 'alges', 'batlow') or matplotlib colormap (e.g., 'viridis')
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

        style = PlotStyle(cmap='batlow')  # Uses palette from PALETTES
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
            'cmap': get_palette_colormap('batlowK'),
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
            # Handle string input - check PALETTES first, then assume matplotlib colormap
            if isinstance(cmap, str):
                if cmap in PALETTES:
                    return get_palette_colormap(cmap)
                else:
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
            # Handle string input - check PALETTES first, then assume matplotlib colormap
            if isinstance(cmap, str):
                if cmap in PALETTES:
                    return get_palette_colormap(cmap)
                else:
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
        """Returns the list of available themes."""
        return list(self.THEMES.keys())
    
    def reset_to_original(self) -> None:
        """Restore the initial matplotlib configuration."""
        plt.rcParams.update(self._original_rcparams)
    
    def get_theme_info(self, theme_name: str):
        """Returns information about a specific theme."""
        if theme_name not in self.THEMES:
            raise ValueError(f"Theme '{theme_name}' not found.")
        return self.THEMES[theme_name].copy()
    
    def __repr__(self):
        """String representation of the object."""
        return f"PlotStyle(theme='{self.theme}', color='{self.color}', cmap='{self.cmap}')"

    def __enter__(self):
        """Context manager support - apply the theme."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support - restores original configuration."""
        self.reset_to_original()

def plot_histogram(data, ax, color='skyblue', alpha = 0.9, rwidth=0.92, hide_empty = True):
    """
    Plots a histogram of the given data using matplotlib
    :param data: The data to visualize. Should be either a 1D array-like object 
                 (list, numpy array, pandas Series) containing numerical values.
    :param ax: Matplotlib Axes object where the histogram will be plotted.
    :param color: Color of the histogram bars. Default assigns 'skyblue'.
    :param alpha: Sets transparency of the histogram bars. Default assigns alpha=0.9 if not specified.
    :param rwidth: Sets relative width of the histogram bars. Default assigns rwidth=0.92 if not specified.
    :param hide_empty: Whether to hide tick labels for empty histogram bins. Default assigns True.
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

def plot_colormap_data(data, ax=None, w=None, h=None, xi_locations=None, griddata=False, title="",
                       figsize = None, dpi = 120, **imshow_args):
    """
    Plots a colormap (heatmap-like visualization) of the given data using matplotlib.

    :param data: The data to visualize. Should be either 2D grid data (if `griddata=True`)
        or a flat array that can be reshaped into a 2D array.
    :param ax: Matplotlib axes object to plot into. If None, a new figure and axes will be created.
    :param w: Width of the image (number of columns). Required if `griddata=False` and `xi_locations` is not provided.
    :param h: Height of the image (number of rows). Required if `griddata=False` and `xi_locations` is not provided.
    :param xi_locations: Coordinates of the data points, used to infer shape (h, w) if not provided explicitly.
    :param griddata: If True, assumes `data` is already in 2D format and transposes it for display.
    :param title: Title to display on the plot (only used if `ax` is None and a new figure is created).
    :param figsize: Width, height of the figure in inches (only used if `ax` is None and a new figure is created).
    :param dpi: The resolution of the figure in dots-per-inch (only used if `ax` is None and a new figure is created). Defaults to 120.
    :param imshow_args: Additional keyword arguments to pass to `imshow()`, such as `cmap`, `vmin`, or `vmax`.

    :raises SpatializeError: If neither `w`/`h` nor `xi_locations` are provided when `griddata` is False,
                             or if reshaping fails due to inconsistent dimensions.
    """
    if griddata:
        im = data.T
    elif w and h:
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
        plotter.set_title(title)

    img = plotter.imshow(im, origin='lower', **imshow_args)
    divider = make_axes_locatable(plotter)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cax.grid(False)
    plt.colorbar(img, orientation='vertical', cax=cax)
    
def plot_colormap_array(data, n_imgs=9, n_cols=3, norm_lims=False, xi_locations=None,
                        reference_map=None, title=None, title_prefix="scenario", seed=None, 
                        figsize=(10, 10), dpi=120, **imshow_args):
    """
    Plots an array of colormap visualizations (heatmaps) for a subset of data columns in a grid layout.

    :param data: The data to visualize. A 2D array where each column represents a separate dataset to plot.
    :param n_imgs: The number of images (subplots) to display from the data. Defaults to 9.
    :param n_cols: The number of columns in the plot grid. Defaults to 3.
    :param norm_lims: If True, normalizes the colormap limits using a reference map. Defaults to False.
    :param xi_locations: Coordinates of the data points. Used for reshaping the data if necessary.
    :param reference_map: A map to use for normalizing the colormap range (`vmin`, `vmax`).
        The min and max of this map are used to adjust the normalization.
    :param title: The title for the entire plot. If `None`, no title will be set.
    :param title_prefix: Prefix for individual subplot titles (e.g., "scenario 1", "scenario 2").
    :param seed: The seed for random number generation. If `None`, a random seed will be generated for reproducibility.
    :param figsize: Width, height of the figure in inches. Defaults to (10, 10).
    :param dpi: The resolution of the figure in dots-per-inch. Defaults to 120.
    :param imshow_args: Additional arguments to pass to the `plot_colormap_data` function, such as `cmap`, `vmin`, or `vmax`.

    :return: A Matplotlib figure object and the seed used for random sampling.
    :rtype: matplotlib.figure.Figure, int

    :raises ValueError: If the number of images to plot exceeds the number of columns in the data.
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
    axes = gs.subplots().flatten()

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
