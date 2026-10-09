"""Posterior analysis of the data: each datum read against the predictive law the other data give.

Cross-validation predicts each datum from the others, with the partitions and the decoder alone, so
the law at a datum carries the evidence the rest of the data give about it, without a variogram or
a model chosen beforehand. :func:`posterior_audit` computes these laws and returns a
:class:`PosteriorAudit`. :func:`cv_sample_pred_posterior` and :class:`PosteriorSampleAnalyzer`, the
names of version 1.2, are kept on top of it.
"""
import random as rd
import warnings
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

import spatialize.gs.esi.aggfunction as af
from spatialize import SpatializeError, logging
from spatialize._parallel import map_chunks
from spatialize._util import signature_overload, per_call, random_seed
from spatialize.empirical import (EmpiricalModel, FittedModelFactory, _loo_target_variance,
                                  _loo_target_skewness)
from spatialize.gs import (lib_spatialize_facade, partitioning_process, local_interpolator as li,
                           with_more_decoders, decoder_arguments)
from spatialize.logging import default_singleton_callback, log_message
from spatialize.viz import PlotStyle


def _default_factory():
    return FittedModelFactory(nan_model_name="ignore", point_model_name="vim", n_components=3,
                              bgm_sample_size=1000, bgm_max_iter=100)


_COMMON = {"k": -1,
           "p_process": partitioning_process.MONDRIAN,
           "data_cond": True,
           "n_partitions": 200,
           "alpha": 0.8,
           "seed": random_seed,
           "folding_seed": random_seed,
           "fitted_model_factory": per_call(_default_factory),
           "callback": default_singleton_callback,
           "best_params_found": None}

_SPECIFIC = with_more_decoders({
    li.IDW: {"exponent": 2.0},
    li.KRIGING: {"model": "spherical", "nugget": 0.5, "range": 50.0, "sill": 0.9},
    li.ADAPTIVE_IDW: {"metric": "mae"},
})


def _with_best_params(kwargs):
    """The arguments with those of ``best_params_found`` in force, except ``n_partitions``."""
    best = kwargs.get("best_params_found")
    if best is None:
        return kwargs
    log_message(logging.logger.debug(f"using best params found: {best}"))
    out = dict(kwargs)
    out.update({key: value for key, value in best.items() if key != "n_partitions"})
    return out


def _cv_members(points, values, queries, kwargs):
    """The cross-validation members at the data, (n, n_partitions), computed through ``run``.

    ``queries`` only take part in the box of the partitions, as in version 1.2."""
    points = np.asarray(points, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    n = points.shape[0]
    k = kwargs["k"]
    method = "loo" if k in (-1, n) else "kfold"
    alpha = kwargs["alpha"]
    if alpha >= 1.0:
        raise ValueError(f"alpha must be < 1 (got {alpha})")
    if kwargs["p_process"] == partitioning_process.VORONOI and not kwargs["data_cond"]:
        alpha = -alpha
    decoder = kwargs["local_interpolator"]
    params = {name: kwargs.get(name) for name in decoder_arguments(decoder)}
    try:
        _, members = lib_spatialize_facade.run(
            points, values, np.asarray(queries, dtype=np.float32), kwargs["p_process"], decoder, params, alpha,
            kwargs["n_partitions"], kwargs["seed"], method=method, k=0 if method == "loo" else k,
            folding_seed=kwargs["folding_seed"], callback=kwargs["callback"])
    except Exception as e:
        raise SpatializeError(e) from e
    return np.asarray(members)


class PosteriorAudit:
    """The predictive law of each datum built from the other data, and its readings.

    Parameters
    ----------
    members : ndarray of shape (n, T)
        The cross-validation members at each datum, one per partition: each datum predicted from the
        other data. NaN where the datum's cell held no other datum (``empty_cells="nan"``).
    points : ndarray of shape (n, d)
        The data locations.
    values : ndarray of shape (n,)
        The data values.
    fitted_model_factory : FittedModelFactory, optional
        The density model fitted to each datum's members (:mod:`spatialize.empirical`). Default: a
        variational Gaussian mixture of three components.
    callback : callable, optional
        Progress and logging callback.

    Attributes
    ----------
    members, points, values
        As given.
    support : ndarray of shape (n,)
        The share of defined members of each datum: the share of partitions in which its law rests
        on other data. A low support marks an isolated datum, whose law rests on few partitions.

    Notes
    -----
    The law of a datum never contains the datum itself, so a value its neighbours do not support
    keeps all of its surprise.
    """

    def __init__(self, members, points, values, fitted_model_factory=None, callback=default_singleton_callback):
        self.members = np.asarray(members, dtype=np.float64)
        self.points = np.asarray(points)
        self.values = np.asarray(values, dtype=np.float64).ravel()
        if self.members.shape[0] != len(self.values):
            raise SpatializeError(f"members has {self.members.shape[0]} rows for {len(self.values)} data")
        self.fitted_model_factory = fitted_model_factory if fitted_model_factory is not None else _default_factory()
        self.callback = callback
        self.support = np.isfinite(self.members).mean(axis=1)
        self._models = {}
        self._targets = None

    def _widening_targets(self):
        f = self.fitted_model_factory
        if self._targets is None:
            # leave-one-out targets: each datum's k nearest OTHER data, so its own value does not
            # deflate its target
            var = _loo_target_variance(self.points, self.values, knn=f.widening_knn) if f.widening else None
            skew = (_loo_target_skewness(self.points, self.values, knn=f.widening_knn)
                    if f.widening == "skew_normal" else None)
            self._targets = (var, skew)
        return self._targets

    def model(self, i):
        """The density model of datum ``i``'s law, fitted to its defined members, or None.

        Parameters
        ----------
        i : int
            Index of the datum.

        Returns
        -------
        EmpiricalModel or None
            None when fewer than two members are defined or the fit fails.
        """
        if i not in self._models:
            sample = self.members[i][np.isfinite(self.members[i])]
            f = self.fitted_model_factory
            var, skew = self._widening_targets()
            model = None
            if sample.size >= 2:
                try:
                    model = EmpiricalModel(sample=sample, fitted_model_factory=f,
                                           target_var=None if var is None else var[i],
                                           target_skew=None if skew is None else skew[i],
                                           seed=None if f.seed is None else f.seed + i)
                except Exception as e:
                    log_message(logging.logger.debug(f"no law for datum {i} (value {self.values[i]}): {e}"))
            self._models[i] = model
        return self._models[i]

    def models(self):
        """The density models of every datum (see :meth:`model`), warning once about the data
        without one.

        Returns
        -------
        list of EmpiricalModel or None
        """
        out = [self.model(i) for i in range(len(self.values))]
        missing = sum(m is None for m in out)
        if missing:
            log_message(logging.logger.warning(
                f"{missing} of {len(out)} data have no fitted law (fewer than two defined members or a "
                f"failed fit); their readings are NaN"))
        return out


class PosteriorSampleAnalyzer(PosteriorAudit):
    """The posterior analysis of version 1.2, on top of :class:`PosteriorAudit`.

    Parameters
    ----------
    cv_post_result : ndarray of shape (n, T)
        The cross-validation members at each datum.
    points : ndarray of shape (n, d)
        The data locations.
    sample_values : ndarray of shape (n,)
        The data values.
    fitted_model_factory : FittedModelFactory
        The density model fitted to each datum's members.
    callback : callable, optional
        Progress and logging callback.

    Attributes
    ----------
    post_result : ndarray
        The members, as ``members``.
    sample_values : ndarray
        The data values, as ``values``.
    emodels : dict of int to EmpiricalModel
        The fitted law of each datum that has one.
    sample_quantiles : dict of int to float
        The value of each datum's cumulative distribution function at the datum, NaN without a law.
    sample_entropy : dict of int to float
        The entropy of each datum's law, NaN without a law.

    Notes
    -----
    Since version 1.3 the law of a datum is fitted to its members alone. Version 1.2 added the datum
    to its own members, which put a kernel on it and capped its surprise.
    """

    def __init__(self, cv_post_result, points, sample_values, fitted_model_factory,
                 callback=default_singleton_callback):
        super().__init__(cv_post_result, points, sample_values, fitted_model_factory, callback)
        self.post_result = cv_post_result
        self.sample_values = sample_values
        models = self.models()
        self.emodels = {i: m for i, m in enumerate(models) if m is not None}
        self.sample_quantiles, self.sample_entropy = {}, {}
        for i, m in enumerate(models):
            q = h = np.nan
            if m is not None:
                try:
                    h, q = float(m.entropy()), float(m.cdf(self.values[i]))
                except Exception as e:
                    log_message(logging.logger.debug(f"no readings for datum {i}: {e}"))
            self.sample_quantiles[i], self.sample_entropy[i] = q, h

    def rank_samples(self, entropy_mass_alphas=[0.5, 0.7, 0.9, 0.99]):
        """Categorizes sample values based on their central entropy intervals.

        A sample is assigned to ``level_j`` if it falls outside the interval of the ``(j+1)``-th
        largest alpha but inside every wider one; ``level_0`` holds the samples outside the widest
        interval (the tails), ``level_k`` (k = number of alphas) those inside every interval.

        Parameters
        ----------
        entropy_mass_alphas : list of float, optional
            The masses of the intervals. Default: ``[0.5, 0.7, 0.9, 0.99]``.

        Returns
        -------
        pandas.DataFrame
            ``value`` and ``category`` (``"level_j"``, or None without a law).

        Notes
        -----
        The samples are ranked on several processes (``joblib``) under the session settings
        ``parallel`` and ``num_threads`` (:mod:`spatialize.session`), when the work is large enough
        to repay starting them. When it runs in parallel, a script should call this method within an
        ``if __name__ == "__main__":`` block.
        """
        alphas_ = sorted(entropy_mass_alphas)
        values, emodels = self.sample_values, self.emodels

        def rows_data(rows):
            idx = list(rows)
            return [values[k] for k in idx], [emodels.get(k) for k in idx]

        def categorize_rows(rows, data):
            vals, models = data
            out = []
            for i, value, emodel in zip(rows, vals, models):
                if emodel is None:
                    out.append(None)
                    continue
                try:
                    cat = len(alphas_)
                    for j, alpha in enumerate(reversed(alphas_)):
                        low, high = emodel.central_entropy_interval(alpha)['interval']
                        if value < low or value > high:
                            cat = j
                            break
                    out.append(f"level_{cat}")
                except Exception as e:
                    log_message(logging.logger.debug(f"error for values[{i}] = {value}: {e}"))
                    out.append(None)
            return out

        categories = map_chunks(categorize_rows, len(values), data_for=rows_data, callback=self.callback)
        log_message(logging.logger.info(
            f"categorized {len(values)} samples into {len(set(c for c in categories if c is not None))} categories."))
        return pd.DataFrame({'value': self.sample_values, 'category': categories})

    def plot_summary(self, theme='alges', color=None, **figargs):
        """Histograms of the values, of the cumulative probabilities of the data in their laws and of
        the entropies of the laws.

        Parameters
        ----------
        theme : str, optional
            Plot theme (:class:`~spatialize.viz.PlotStyle`). Default: ``'alges'``.
        color : str, optional
            Histogram colour. Default: the theme's.
        **figargs
            Passed to :func:`matplotlib.pyplot.subplots`, e.g. ``figsize=(12, 4)``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        with PlotStyle(theme=theme, color=color) as style:
            fig, ax = plt.subplots(1, 3, **figargs)
            fig.suptitle("Posterior Sample Analysis")
            fig.subplots_adjust(wspace=0.3)
            for a, data, title in ((ax[0], self.values, "Value"),
                                   (ax[1], list(self.sample_quantiles.values()), "Percentiles"),
                                   (ax[2], list(self.sample_entropy.values()), "Entropy")):
                data = np.asarray(data, dtype=float)
                a.hist(data[np.isfinite(data)], 25, density=True, histtype='stepfilled', alpha=0.8,
                       color=style.color, zorder=3)
                a.set_title(title)
        return fig

    def quick_plot_models(self, n_imgs=6, n_cols=3, seed=42, theme='alges', cmap=None, **figargs):
        """A grid of the laws of randomly chosen data: histogram of the members, density and scaled
        cumulative distribution function.

        Parameters
        ----------
        n_imgs : int, optional
            Number of data shown. Default: 6.
        n_cols : int, optional
            Number of columns of the grid. Default: 3.
        seed : int, optional
            Seed of the choice of data. Default: 42.
        theme : str, optional
            Plot theme. Default: ``'alges'``.
        cmap : str or Colormap, optional
            Colours of the panels. Default: the theme's.
        **figargs
            Passed to :func:`matplotlib.pyplot.subplots`.

        Returns
        -------
        matplotlib.figure.Figure or None
            None when no datum has a law.
        """
        available = sorted(self.emodels)
        n_imgs = min(n_imgs, len(available))
        if not n_imgs:
            log_message(logging.logger.warning("No empirical models available to plot in quick_plot_models."))
            return None
        n_rows = -(-n_imgs // n_cols)
        state = rd.getstate()
        if seed is not None:
            rd.seed(seed)
        idx = rd.sample(available, k=n_imgs)
        rd.setstate(state)
        return plot_histogram_grid_with_pdf_cdf(self.post_result, idx, self.emodels, n_rows, n_cols, bins=25,
                                                theme=theme, cmap=cmap, **figargs)

    def plot_ranking(self, samples_ranking, theme='alges', color=None, cmap=None, figsize=(11, 6)):
        """The counts of each category and the data coloured by category, on their first two
        coordinates.

        Parameters
        ----------
        samples_ranking : pandas.DataFrame
            The output of :meth:`rank_samples`.
        theme : str, optional
            Plot theme. Default: ``'alges'``.
        color : str, optional
            Bar colour. Default: the theme's.
        cmap : str or Colormap, optional
            Colours of the categories. Default: the theme's.
        figsize : tuple, optional
            Figure size. Default: ``(11, 6)``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        with PlotStyle(theme=theme, color=color, cmap=cmap) as style:
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            categories = samples_ranking["category"]
            counts = categories.value_counts().sort_index()
            ax[0].bar(counts.index.astype(str), counts.values, color=style.color, zorder=3)
            ax[0].set_title("Categories")
            ax[0].tick_params(axis='x', rotation=45)

            names = sorted(set(str(c) for c in categories if c is not None))
            ax[1].set_title("Categorized Samples")
            if not names:
                log_message(logging.logger.warning("No valid categories to plot in plot_ranking."))
                fig.tight_layout()
                return fig
            number = {c: i for i, c in enumerate(names)}
            nums = np.array([number.get(str(c), -1) if c is not None else -1 for c in categories])
            ok = nums >= 0
            base = matplotlib.colormaps[style.cmap] if isinstance(style.cmap, str) else style.cmap
            cat_cmap = ListedColormap(base.resampled(max(len(names), 2))(np.linspace(0, 1, len(names))))
            pts = np.asarray(self.points, dtype=float)
            y = pts[ok, 1] if pts.shape[1] > 1 else np.zeros(ok.sum())
            sc = ax[1].scatter(pts[ok, 0], y, c=nums[ok], cmap=cat_cmap, s=30, edgecolor='white',
                               linewidth=0.5, zorder=3, vmin=-0.5, vmax=len(names) - 0.5)
            cbar = plt.colorbar(sc, ax=ax[1], ticks=list(range(len(names))), orientation='vertical')
            cbar.set_ticklabels(names)
            ax[1].set_aspect('equal', adjustable='box')
            ax[1].set_xlabel("X")
            ax[1].set_ylabel("Y")
            fig.tight_layout()
        return fig


@signature_overload(pivot_arg=("local_interpolator", li.IDW, "local interpolator"),
                    common_args=dict(_COMMON),
                    specific_args=_SPECIFIC)
def posterior_audit(points, values, **kwargs):
    """The predictive law of each datum built from the other data.

    Each datum is predicted from the other data by cross-validation (leave-one-out, or k-fold with
    ``k``) with the partitions and the decoder, so its law carries the evidence the rest of the data
    give about it, with no variogram or model chosen beforehand.

    Parameters
    ----------
    points : array_like of shape (n, d)
        The data locations.
    values : array_like of shape (n,)
        The data values.
    local_interpolator : str, optional
        The decoder, any of the catalogue (:func:`~spatialize.gs.esi.esi_nongriddata`), with its
        parameters as keyword arguments. Default: ``"idw"``.
    k : int, optional
        ``-1`` (or n) for leave-one-out, otherwise the number of folds. Default: -1.
    p_process : {"mondrian", "mondrian-raw", "voronoi"}, optional
        The partition process. Default: ``"mondrian"``.
    data_cond : bool, optional
        Voronoi only: nuclei among the data (True) or uniform in the box. Default: True.
    n_partitions : int, optional
        The number of partitions, the members of each law. Default: 200.
    alpha : float, optional
        The granularity of the partitions. Default: 0.8.
    seed, folding_seed : int, optional
        Seeds of the partitions and of the folds. Default: drawn at random.
    fitted_model_factory : FittedModelFactory, optional
        The density model of each law. Default: a variational Gaussian mixture of three components.
    best_params_found : dict, optional
        The output of a search's ``best_result()``; its keys override the arguments, except
        ``n_partitions``. The dict is not modified. Default: None.
    callback : callable, optional
        Progress and logging callback.

    Returns
    -------
    PosteriorAudit

    Notes
    -----
    The partitions are drawn on the session domain (:mod:`spatialize.session`), or on the box of the
    data without one. Cells without other data follow the session setting ``empty_cells``: under
    ``"nan"``, the default, an isolated datum's law rests on the partitions in which its cell holds
    other data, the share :attr:`PosteriorAudit.support` reports.
    """
    kwargs = _with_best_params(kwargs)
    members = _cv_members(points, values, points, kwargs)
    return PosteriorAudit(members, points, values, kwargs["fitted_model_factory"], callback=kwargs["callback"])


@signature_overload(pivot_arg=("local_interpolator", li.IDW, "local interpolator"),
                    common_args=dict(_COMMON, griddata=False, agg_function=af.mean),
                    specific_args=_SPECIFIC)
def cv_sample_pred_posterior(points, values, xi, **kwargs):
    """The posterior analysis of version 1.2: the law of each datum built from the other data.

    Parameters
    ----------
    points : ndarray of shape (n, d)
        The data locations.
    values : ndarray of shape (n,)
        The data values.
    xi : ndarray or tuple
        Locations that only enlarge the box the partitions are drawn on, as in version 1.2.
        :func:`posterior_audit` has no such argument, its box being the session domain or that of
        the data.
    **kwargs
        As :func:`posterior_audit`; ``griddata`` and ``agg_function`` are accepted and ignored.

    Returns
    -------
    PosteriorSampleAnalyzer

    Raises
    ------
    SpatializeError
        If the cross-validation fails.
    """
    kwargs = _with_best_params(kwargs)
    queries = deepcopy(xi) if isinstance(xi, tuple) else np.asarray(xi).copy()
    members = _cv_members(points, values, queries, kwargs)
    log_message(logging.logger.info(f"using fitted model factory: {kwargs['fitted_model_factory']}"))
    return PosteriorSampleAnalyzer(members, points, values, kwargs['fitted_model_factory'],
                                   callback=kwargs['callback'])


def plot_histogram_grid_with_pdf_cdf(r, data_indices, emodels, n_rows, n_cols, bins=25, figsize=(15, 10),
                                     theme='alges', cmap=None):
    """A grid of the laws of the given data: histogram of the fitted data, density and scaled
    cumulative distribution function.

    Parameters
    ----------
    r : ndarray of shape (n, T)
        The members of each datum.
    data_indices : list of int
        The data shown.
    emodels : dict of int to EmpiricalModel, or list
        The fitted law of each datum.
    n_rows, n_cols : int
        The shape of the grid.
    bins : int, optional
        Number of bins of the histograms. Default: 25.
    figsize : tuple, optional
        Figure size. Default: ``(15, 10)``.
    theme : str, optional
        Plot theme. Default: ``'alges'``.
    cmap : str or Colormap, optional
        Colours of the panels. Default: the theme's.

    Returns
    -------
    matplotlib.figure.Figure
    """
    with PlotStyle(theme=theme, cmap=cmap) as style:
        base = matplotlib.colormaps[style.cmap] if isinstance(style.cmap, str) else style.cmap
        panel_colors = base(np.linspace(0.15, 0.85, max(n_rows * n_cols, 2)))
        pdf_color = plt.rcParams['text.color']
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            if i >= len(data_indices):
                ax.axis('off')
                continue
            idx = data_indices[i]
            try:
                emodel = emodels[idx]
            except (KeyError, IndexError, TypeError):
                log_message(logging.logger.warning(f"Empirical model for index {idx} not found."))
                ax.axis('off')
                continue
            data = getattr(emodel, "data_", None)
            if data is None:
                data = np.asarray(r[idx, :])
                data = data[np.isfinite(data)]
            ax.hist(data, bins=bins, density=True, histtype='stepfilled', alpha=0.8,
                    color=panel_colors[i % len(panel_colors)], edgecolor='black', zorder=2)
            ax.plot(emodel.x_, emodel.pdf_, '-', color=pdf_color, label="PDF", zorder=3)
            lo, hi = ax.get_ylim()
            ax.plot(emodel.x_, emodel.cdf_ * (hi - lo) + lo, '-b', label="CDF (scaled)", zorder=3)
            ax.set_title(f'Sample {idx}', fontsize=10)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.legend(fontsize=6)
        fig.tight_layout()
    return fig
