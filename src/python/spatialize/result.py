import numpy as np
from matplotlib import pyplot as plt

from spatialize import SpatializeError, in_notebook
from spatialize.viz import plot_colormap_data, plot_histogram, PlotStyle

from typing import Optional, Dict, Any


class GridSearchResult:
    def __init__(self, search_result_data):
        self.search_result_data = search_result_data

        data = self.search_result_data
        self.cv_error = data[['cv_error']]
        min_error = self.cv_error.min()['cv_error']
        self.best_params = data[data.cv_error <= min_error]

    def plot_cv_error(self,
                      theme: Optional[str] = 'alges',
                      color: Optional[str] = None,
                      fig_args: Optional[Dict[str, Any]] = None
                      ):
        """
        It shows a graph of the cross-validation errors of the hyperparameter
        search process. The graph has two components: the first is the error histogram,
        and the second is the error level for each of the estimation scenarios generated
        by the gridded parameter search.

        :param fig_args: Dictionary with figure configuration for plt.subplots().
            Default assigns figsize=(10, 4) if not specified.
        :param subplot_args: Dictionary with subplot configuration for
            plt.subplots_adjust(). Default assigns wspace=0.45 if not specified.
        :param theme: Theme name. Available: 'whitegrid', 'darkgrid', 'white', 'dark',
            'alges', 'minimal', 'publication'
        :param color: Color for the plots. If None, uses theme default or 'skyblue'
        :return: Tuple with matplotlib figure and tuple of axes (fig, (ax1, ax2))
        :raises ValueError: If the specified theme does not exist.
        """
        # Default values for figsize if not specified
        if fig_args is None:
            fig_args = {'figsize': (10, 4)}
        elif 'figsize' not in fig_args:
            fig_args['figsize'] = (10, 4)

        with PlotStyle(theme=theme, color=color) as style:
            fig, (ax1, ax2) = plt.subplots(1, 2, **fig_args)
            plt.subplots_adjust(wspace=0.45)
            fig.suptitle("Cross Validation Error")

            plot_histogram(self.cv_error, ax1, style.color)

            self.cv_error.plot(kind='line',
                               ax=ax2,
                               y='cv_error',
                               xlabel="Search result data index",
                               ylabel="Error",
                               color=style.color,
                               lw=2,
                               legend=False)
            
            return fig, (ax1, ax2)

    def best_result(self, **kwargs):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


class EstimationResult:
    def __init__(self, estimation, griddata=False, original_shape=None, xi=None):
        self._estimation = estimation
        self.griddata = griddata
        self.original_shape = original_shape
        self._xi = xi

    def estimation(self):
        """
        Returns the estimated values at locations `xi` by aggregating all ESI samples
        using the aggregation function provided in the `agg_function` argument (in both
        function :func:`esi_griddata` and :func:`esi_nongriddata`). This estimate can be changed
        using another aggregation function with the :func:`re_estimate` method of this same class.

        Returns
        =======
        estimation : numpy.ndarray
            An array of dimension $N_{x^*}$, for non-gridded data, and of dimension $d_1 \times d_2$
            for gridded data -- remember that, in this case, $d_1 \times d_2 = N_{x^*}$
        """
        if self.griddata:
            return self._estimation.reshape(self.original_shape)
        else:
            return self._estimation

    def _get_extent(self):
        """Returns [x_min, x_max, y_min, y_max] for 2D spatial data, or None otherwise."""
        if self._xi is None:
            return None
        if self.griddata:
            # xi may be a tuple/list of arrays or an ndarray; use len() for dimension count
            if len(self._xi) != 2:
                return None
            x_min, x_max = self._xi[0].min(), self._xi[0].max()
            y_min, y_max = self._xi[1].min(), self._xi[1].max()
        else:
            if self._xi.shape[-1] != 2:
                return None
            x_min, x_max = self._xi[:, 0].min(), self._xi[:, 0].max()
            y_min, y_max = self._xi[:, 1].min(), self._xi[:, 1].max()
        return [x_min, x_max, y_min, y_max]

    def plot_estimation(self, ax=None, w=None, h=None, theme='alges', cmap=None, **imshow_args):
        """
        Plots the estimation using `matplotlib`.

        Parameters
        ----------
        ax :  (`matplotlib.axes.Axes`, optional)
            The `Axes` object to render the plot on. If `None`, a new `Axes` object is created.
        w : (int, optional)
            The width of the image (if the data is reshaped).
        h : (int, optional)
            The height of the image (if the data is reshaped).
        theme : (str, optional)
            Theme name. Available: 'whitegrid', 'darkgrid', 'white', 'dark',
            'alges', 'minimal', 'publication'
        cmap : (str, optional)
            Colormap for the plot. If None, uses theme default or 'coolwarm'
        **imshow_args : (optional)
            Additional keyword arguments passed to the `_plot_data` function (e.g., vmin, vmax, alpha).

        """
        plot_imshow_args = imshow_args.copy()
        if not cmap:
            cmap = plot_imshow_args.pop('cmap', None)
        if 'extent' not in plot_imshow_args:
            extent = self._get_extent()
            if extent is not None:
                plot_imshow_args['extent'] = extent

        with PlotStyle(theme=theme, cmap=cmap) as style:
            self._plot_data(self.estimation(), ax, w, h, cmap = style.cmap, **plot_imshow_args)

    def _plot_data(self, data, ax=None, w=None, h=None, **imshow_args):
        plot_colormap_data(data, ax=ax, w=w, h=h, xi_locations=self._xi, griddata=self.griddata, **imshow_args)

    def quick_plot(self, w=None, h=None, **imshow_args):
        """
        Quickly plots the estimation using `matplotlib`.

        Parameters
        ----------
        w : (int, optional)
            The width of the image (if the data is reshaped).
        h : (int, optional)
            The height of the image (if the data is reshaped).
        **imshow_args : (optional)
            Additional keyword arguments passed to the figure creation (e.g., DPI, figure size).
        """
        if self._xi is not None:
            if self.griddata:
                if len(self._xi) > 2:
                    raise SpatializeError("quick_plot() for 3D+ data is not supported")
            else:
                if self._xi.shape[1] > 2:
                    raise SpatializeError("quick_plot() for 3D data is not supported")

        fig = plt.figure(dpi=150, **imshow_args)
        gs = fig.add_gridspec(1, 1, wspace=0.45)
        ax = gs.subplots()

        ax.set_title('Estimation')
        self.plot_estimation(ax, w=w, h=h)
        ax.set_aspect('equal')

        if not in_notebook():
            return fig

    def __repr__(self):
        min, max = np.nanmin(self.estimation()), np.nanmax(self.estimation())
        m, s, med = np.nanmean(self.estimation()), np.nanstd(self.estimation()), np.nanmedian(self.estimation())
        msg = (f"estimation results: \n"
               f"  minimum: {min:.3f}, maximum: {max:.3f}\n"
               f"  mean: {m:.2f}, std dev: {s:.2f}, median: {med:.2f}\n"
               f"to display the result, use the method ‘quick_plot()’.\n")
        return msg
