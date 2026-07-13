import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from joblib import Parallel, delayed
from multiprocessing import Manager
import threading
import time

from copy import deepcopy
from spatialize.gs import lib_spatialize_facade, partitioning_process, local_interpolator as li
import spatialize.gs.esi.aggfunction as af
from spatialize.gs.esi._main import build_arg_list
from spatialize._util import signature_overload
from spatialize.logging import default_singleton_callback, log_message
from spatialize import SpatializeError, logging  # Assuming 'logging' here refers to your custom logging
from spatialize.empirical import (EmpiricalModel, FittedModelFactory, _loo_target_variance,
                                   _loo_target_skewness)


class PosteriorSampleAnalyzer:
    """
    Analyzes results from cross-validated posterior samples.

    This class takes the output of a cross-validation process, along with
    the original sample points and values, to compute empirical models,
    quantiles, and entropy for each sample. It also provides methods for
    ranking samples based on entropy and for visualizing the results.
    """

    def __init__(self, cv_post_result, points, sample_values, fitted_model_factory,
                 callback=default_singleton_callback):
        """
        Initialize the PosteriorSampleAnalyzer.

        :param cv_post_result: The posterior results from a cross-validation process.
                               Expected to be a 2D numpy array where rows correspond
                               to samples and columns to posterior draws.
        :type cv_post_result: numpy.ndarray
        :param points: Coordinates of the sample points.
        :type points: numpy.ndarray
        :param sample_values: The true values at the sample points.
        :type sample_values: numpy.ndarray
        :param fitted_model_factory: A factory object to create fitted empirical models.
        :type fitted_model_factory: empirical.FittedModelFactory
        :param callback: Optional callback function for logging or progress updates.
                         Defaults to `default_singleton_callback`.
        :type callback: callable, optional
        """
        self.post_result = cv_post_result
        self.points = points
        self.sample_values = sample_values
        self.fitted_model_factory = fitted_model_factory
        self.callback = callback

        self.emodels = {}
        self.sample_quantiles = {}
        self.sample_entropy = {}

        # ensemble widening needs a target local variance (and, for widening="skew_normal",
        # a target local skewness); this is a leave-one-out CV result, so points and
        # sample_values coincide -- use the self-excluding k-NN variant to avoid each point's
        # zero-distance match to itself deflating its target
        target_var_arr = (
            _loo_target_variance(points, sample_values) if fitted_model_factory.widening else None
        )
        target_skew_arr = (
            _loo_target_skewness(points, sample_values)
            if fitted_model_factory.widening == "skew_normal" else None
        )

        for i in range(len(self.sample_values)):
            data = np.append(self.post_result[i, :], self.sample_values[i])
            target_var = target_var_arr[i] if target_var_arr is not None else None
            target_skew = target_skew_arr[i] if target_skew_arr is not None else None
            emodel = EmpiricalModel(sample=data, fitted_model_factory=fitted_model_factory,
                                    target_var=target_var, target_skew=target_skew)
            self.emodels[i] = emodel
            try:
                h = emodel.entropy()
                self.sample_entropy[i] = h
                p = emodel.cdf(self.sample_values[i])
                self.sample_quantiles[i] = p
            except Exception as e:
                # Use your existing logging mechanism
                log_message(logging.logger.debug(f"error for values[{i}] = {self.sample_values[i]}: {e}"))
                continue

    def rank_samples(self, entropy_mass_alphas=[0.5, 0.7, 0.9, 0.99], n_jobs=-1):
        """
        Categorizes sample values based on their central entropy intervals.

        The categories are determined by the specified entropy masses (alphas),
        which define the width of the intervals. A smaller alpha corresponds to a
        narrower interval (more central mass), and samples falling outside wider
        intervals are considered more "certain" (i.e., they are in the tails of
        their posterior distribution).

        The categorization logic is as follows:
        A sample is assigned to `level_j` if it falls outside the interval
        defined by `alphas_[-(j+1)]` but inside all wider intervals (those with
        larger alpha values). `level_0` represents the most certain samples
        (outside the widest interval), and `level_k` (where k = len(alphas))
        represents the most uncertain samples (inside the narrowest interval).

        For example, with default `entropy_mass_alphas = [0.5, 0.7, 0.9, 0.99]`:

        +----------+---------------------------------------+--------------------------+
        | Category | Interval Match                        | Interpretation           |
        +==========+=======================================+==========================+
        | level_0  | outside 99% interval                  | most **certain** (tail)  |
        +----------+---------------------------------------+--------------------------+
        | level_1  | outside 90% interval, inside 99%      | more certain             |
        +----------+---------------------------------------+--------------------------+
        | level_2  | outside 70% interval, inside 90%      | moderate certainty       |
        +----------+---------------------------------------+--------------------------+
        | level_3  | outside 50% interval, inside 70%      | less certain             |
        +----------+---------------------------------------+--------------------------+
        | level_4  | inside 50% interval                   | most **uncertain**       |
        +----------+---------------------------------------+--------------------------+

        :param entropy_mass_alphas: List of alpha values (between 0 and 1)
                                    defining the central entropy mass for intervals.
                                    Defaults to `[0.5, 0.7, 0.9, 0.99]`.
        :type entropy_mass_alphas: list[float], optional
        :param n_jobs: Number of parallel jobs to run for categorization.
                       -1 means using all available cores. 1 means serial execution.
                       Defaults to -1. When using ``n_jobs != 1``, this function must be called within a
                       ``if __name__ == "__main__":`` block to avoid multiprocessing issues
                       (especially on Windows and macOS).
        :type n_jobs: int, optional
        :return: A DataFrame with 'value' (original sample value) and
                 'category' (e.g., "level_0") columns.
        :rtype: pandas.DataFrame
        """
        alphas_ = sorted(entropy_mass_alphas)  # narrowest to widest intervals
        categories_results = []  # Using a temporary list to store results before assigning to self or returning

        values = self.sample_values
        callback = self.callback

        def run_serial():
            local_categories = []
            callback(logging.progress.init(len(values), 1))
            for i in range(len(values)):
                emodel = self.emodels[i]
                try:
                    # default: most uncertain category (inside all intervals)
                    cat = len(alphas_)
                    # check from widest to narrowest (reversed alphas)
                    for j, alpha in enumerate(reversed(alphas_)):
                        cei = emodel.central_entropy_interval(alpha)
                        low, high = cei['interval']
                        if values[i] < low or values[i] > high:
                            # falls outside this interval → more certain
                            cat = j
                            break  # stop at most certain matching category

                    local_categories.append(f"level_{cat}")  # Corrected category naming
                except Exception as e:
                    log_message(logging.logger.debug(f"error for values[{i}] = {values[i]}: {e}"))
                    local_categories.append(None)  # Append None or some error marker
                    continue
                callback(logging.progress.inform())
            callback(logging.progress.stop())
            return local_categories

        # just for parallel execution ----------------------------------------------------------------
        def categorize_single_sample(idx, value_item, emodel_item, alphas_list, progress_q):
            try:
                cat_val = len(alphas_list)
                for j_idx, alpha_val in enumerate(reversed(alphas_list)):
                    cei_val = emodel_item.central_entropy_interval(alpha_val)
                    low_val, high_val = cei_val['interval']
                    if value_item < low_val or value_item > high_val:
                        cat_val = j_idx
                        break
                if progress_q:  # Check if progress_queue is provided (it will be)
                    progress_q.put(1)
                return idx, f"level_{cat_val}"  # Corrected category naming
            except Exception as e_inner:
                log_message(logging.logger.debug(f"error for values[{idx}] = {value_item}: {e_inner}"))
                if progress_q:
                    progress_q.put(1)  # Still signal progress even on error
                return idx, None  # Return None or an error marker for this sample

        def run_parallel():
            # Progress monitor thread remains similar
            def progress_monitor(queue, total, cb):
                count = 0
                cb(logging.progress.init(total, 1))  # Init progress here for monitor
                while count < total:
                    queue.get()  # Blocks until an item is available
                    count += 1
                    cb(logging.progress.inform())
                cb(logging.progress.stop())

            # Main parallel execution logic
            # callback(logging.progress.init(len(values), 1)) # Moved to progress_monitor

            with Manager() as manager:
                progress_queue = manager.Queue()

                monitor_thread = threading.Thread(
                    target=progress_monitor,
                    args=(progress_queue, len(values), callback),
                    daemon=True
                )
                monitor_thread.start()

                results = Parallel(n_jobs=n_jobs, backend='loky')(  # Use passed n_jobs
                    delayed(categorize_single_sample)(i, values[i], self.emodels[i], alphas_, progress_queue)
                    for i in range(len(values))
                )

                monitor_thread.join()

            # Sort results by original index and extract categories
            # Handle potential None values from errors
            sorted_results = sorted(results, key=lambda x: x[0])
            final_categories = [res[1] for res in sorted_results]
            return final_categories
            # callback(logging.progress.stop()) # Moved to progress_monitor
            # return categories # This was assigned inside run_parallel_with_progress which is now inline

        if n_jobs == 1:
            categories_results = run_serial()
        else:
            categories_results = run_parallel()

        log_message(logging.logger.info(
            f"categorized {len(values)} samples into {len(set(cat for cat in categories_results if cat is not None))} categories."))

        return pd.DataFrame({'value': self.sample_values, 'category': categories_results})

    def plot_summary(self, **figargs):
        """
        Plots a summary of the posterior sample analysis.

        This creates a figure with three subplots:
        1. Histogram of the original sample values.
        2. Histogram of the calculated sample percentiles (CDFs).
        3. Histogram of the calculated sample entropies.

        :param figargs: Keyword arguments to be passed to `matplotlib.pyplot.subplots`.
                        For example, `figsize=(12, 4)`.
        :type figargs: dict, optional
        """
        fig, ax = plt.subplots(1, 3, **figargs)
        fig.suptitle("Posterior Sample Analysis")
        fig.subplots_adjust(wspace=0.5)

        ax[0].hist(self.sample_values, 25, density=True, histtype='stepfilled', alpha=0.3)
        ax[0].set_title("Value")

        ax[1].hist(list(self.sample_quantiles.values()), 25, density=True, histtype='stepfilled',
                   alpha=0.3)  # Ensure it's a list for hist
        ax[1].set_title("Percentiles")

        ax[2].hist(list(self.sample_entropy.values()), 25, density=True, histtype='stepfilled',
                   alpha=0.3)  # Ensure it's a list for hist
        ax[2].set_title("Entropy")

        # Consider plt.show() or returning fig if used in non-interactive environments
        # plt.show()

    def quick_plot_models(self, n_imgs=6, n_cols=3, seed=42, **figargs):
        """
        Plots a grid of individual empirical model fits for a random sample of data points.

        For each selected data point, it plots the histogram of its posterior samples,
        the Probability Density Function (PDF), and the Cumulative Distribution
        Function (CDF) derived from its empirical model.

        :param n_imgs: Number of random samples/images to plot. Defaults to 6.
                       If greater than the total number of samples, it will be capped.
        :type n_imgs: int, optional
        :param n_cols: Number of columns in the plot grid. Defaults to 3.
        :type n_cols: int, optional
        :param seed: Seed for the random number generator to ensure reproducibility
                     of sample selection. Defaults to 42. If None, a random seed is used.
        :type seed: int, optional
        :param figargs: Keyword arguments passed to `matplotlib.pyplot.subplots`
                        when creating the figure.
        :type figargs: dict, optional
        """
        r = self.post_result
        values = self.sample_values

        if n_imgs > values.shape[0]:
            n_imgs = values.shape[0]

        n_rows = n_imgs // n_cols if n_imgs % n_cols == 0 else (n_imgs // n_cols + 1)

        # Handle seeding for reproducibility
        current_random_state = rd.getstate()  # Save current state
        if seed is not None:
            rd.seed(seed)

        # random sample of colormap images to plot
        # Ensure emodels keys match the indices used (0 to len(values)-1)
        available_indices = [i for i in range(values.shape[0]) if i in self.emodels]
        if n_imgs > len(available_indices):
            n_imgs = len(available_indices)  # Cap n_imgs by available models

        if not available_indices:
            log_message(logging.logger.warning("No empirical models available to plot in quick_plot_models."))
            rd.setstate(current_random_state)  # Restore random state
            return

        hist_idx = rd.sample(available_indices, k=n_imgs)
        rd.setstate(current_random_state)  # Restore random state

        plot_histogram_grid_with_pdf_cdf(r, hist_idx, self.emodels, n_rows, n_cols,
                                         bins=25, **figargs)

    def plot_ranking(self, samples_ranking, figsize=(11, 6)):
        """
        Plots the sample ranking results.

        This creates a figure with two subplots:
        1. A bar chart showing the count of samples in each category.
        2. A scatter plot of the sample points, colored by their assigned category.

        :param samples_ranking: DataFrame containing 'value' and 'category' columns,
                                as returned by `rank_samples`.
        :type samples_ranking: pandas.DataFrame
        :param figsize: Size of the figure for plotting. Defaults to (11, 6).
        :type figsize: tuple, optional
        """
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        categories_series = samples_ranking["category"]  # This is a pandas Series
        # For bar plot, count occurrences of each category
        category_counts = categories_series.value_counts().sort_index()

        ax[0].bar(category_counts.index.astype(str), category_counts.values)
        ax[0].set_title("Categories")
        ax[0].tick_params(axis='x', rotation=45)  # Rotate labels if they overlap

        # Scatter plot part
        # Ensure categories are strings for consistent processing
        categories_str = categories_series.astype(str).values
        unique_cats = sorted(
            list(set(c for c in categories_str if c != 'None')))  # Exclude 'None' if it's an error marker

        if not unique_cats:
            log_message(logging.logger.warning("No valid categories to plot in plot_ranking scatter plot."))
            ax[1].set_title("Categorized Samples (No data)")
            plt.tight_layout()
            # plt.show() # Depending on usage
            return

        cat_to_num = {cat: i for i, cat in enumerate(unique_cats)}

        # Map categories to numbers, handling potential 'None' or unmapped
        category_nums = np.array([cat_to_num.get(str(cat), -1) for cat in categories_series.values])

        # Filter out points where category was None or unmapped (-1)
        valid_indices = (category_nums != -1)

        if not np.any(valid_indices):
            log_message(logging.logger.warning("All categories are unmapped or None in plot_ranking."))
            ax[1].set_title("Categorized Samples (No valid data)")
            plt.tight_layout()
            # plt.show()
            return

        points_to_plot = self.points[valid_indices]
        category_nums_to_plot = category_nums[valid_indices]

        # Check array lengths
        if len(points_to_plot) != len(category_nums_to_plot):  # Should not happen if logic is correct
            log_message(logging.logger.error("Mismatch between points and categories after filtering!"))
            # Fallback or raise error
            plt.tight_layout()
            return

        # Create custom colormap
        # cmap = ListedColormap(plt.cm.plasma(np.linspace(0, 1, len(unique_cats))))
        # Use a robust colormap if plt.cm.plasma is not always available or for better distinction
        try:
            cmap_colors = plt.cm.get_cmap('viridis', len(unique_cats))
        except ValueError:  # Fallback if len(unique_cats) is 0 or 1
            cmap_colors = plt.cm.get_cmap('viridis', 2)  # Default to at least 2 colors

        cmap = ListedColormap(cmap_colors(np.linspace(0, 1, len(unique_cats))))

        # Plot
        if points_to_plot.shape[0] > 0:  # Check if there's anything to plot
            sc = ax[1].scatter(points_to_plot[:, 0], points_to_plot[:, 1],
                               c=category_nums_to_plot, cmap=cmap, s=30, edgecolor='none')
            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax[1], ticks=list(range(len(unique_cats))), orientation='vertical')
            cbar.set_ticklabels(unique_cats)  # Set colorbar labels to category names
        else:
            log_message(logging.logger.info("No points to display in categorized scatter plot."))

        # Set the aspect ratio to 1:1 (equal axes)
        ax[1].set_aspect('equal', adjustable='box')
        ax[1].set_title("Categorized Samples")
        ax[1].set_xlabel("X")
        ax[1].set_ylabel("Y")
        # plt.grid(True) # grid can sometimes make scatter plots busy
        plt.tight_layout()
        # plt.show() # Depending on usage


@signature_overload(pivot_arg=("local_interpolator", li.IDW, "local interpolator"),
                    common_args={"k": -1,
                                 "griddata": False,
                                 "p_process": partitioning_process.MONDRIAN,  # partitioning process
                                 "data_cond": True,  # whether to condition the partitioning process on samples
                                 # -- valid only when ‘p_process’ is ‘voronoi’.
                                 "n_partitions": 200,
                                 "alpha": 0.8,
                                 "agg_function": af.mean,
                                 "seed": np.random.randint(1000, 10000),
                                 "folding_seed": np.random.randint(1000, 10000),
                                 "fitted_model_factory": FittedModelFactory(
                                     nan_model_name="ignore",
                                     point_model_name="vim", n_components=3,
                                     bgm_sample_size=1000, bgm_max_iter=100
                                 ),
                                 "callback": default_singleton_callback,
                                 "best_params_found": None
                                 },
                    specific_args={
                        li.IDW: {"exponent": 2.0},
                        li.KRIGING: {"model": "spherical",
                                     "nugget": 0.5,
                                     "range": 50.0,
                                     "sill": 0.9},
                        li.ADAPTIVE_IDW: {}
                    })
def cv_sample_pred_posterior(points, values, xi, **kwargs):
    """
    Performs cross-validation for sample prediction and generates posterior distributions.

    This function utilizes a spatialization library (`lib_spatialize_facade`)
    to perform cross-validation (either k-fold or leave-one-out) on the provided
    sample points and values. It then uses the results to initialize and return
    a `PosteriorSampleAnalizer` object.

    The specific behavior of the cross-validation, including the local interpolator
    (e.g., IDW, Kriging), partitioning process, and aggregation functions,
    is controlled by the `kwargs` and the `@signature_overload` decorator.

    :param points: Coordinates of the sample points (e.g., [[x1,y1], [x2,y2], ...]).
    :type points: numpy.ndarray
    :param values: Observed values at each sample point.
    :type values: numpy.ndarray
    :param xi: Prediction locations or configuration for prediction.
               If tuple, assumed to be deepcopied. Otherwise, copied.
    :type xi: object or tuple
    :param \\**kwargs: Keyword arguments that control the cross-validation and
                      posterior analysis. These are largely defined by the
                      `@signature_overload` decorator and can include:
                      `local_interpolator`, `k` (for k-fold), `p_process`,
                      `n_partitions`, `fitted_model_factory`, `callback`, etc.
                      See the decorator for default values and specific options.
    :type \\**kwargs: dict
    :raises SpatializeError: If an error occurs during the underlying
                             spatialization cross-validation process.
    :return: An analyzer object containing the posterior distributions and
             tools for their analysis.
    :rtype: PosteriorSampleAnalizer
    """
    method, k = "kfold", kwargs["k"]
    if k == points.shape[0] or k == -1:
        method = "loo"

    log_message(logging.logger.debug('calling libspatialize'))

    if kwargs.get("best_params_found") is not None:  # Use .get for safer access
        try:
            log_message(logging.logger.debug(f"best number of partitions found: "
                                             f"{kwargs['best_params_found']['n_partitions']}"))
            # It's generally safer not to delete from kwargs if it's passed around,
            # but if this is intended, it's fine.
            # del kwargs["best_params_found"]["n_partitions"]
        except KeyError:
            pass  # n_partitions might not be in best_params_found
        log_message(logging.logger.debug(f"using best params found: {kwargs['best_params_found']}"))
        for param_key, param_val in kwargs["best_params_found"].items():
            # Only overwrite if not 'n_partitions' or if explicitly allowed to be overwritten
            # The original code overwrites n_partitions from best_params_found if present,
            # then potentially again from kwargs["n_partitions"] if 'n_partitions' was
            # not deleted from best_params_found.
            # The logic here seems to be that `n_partitions` in `kwargs` takes precedence.
            if param_key != "n_partitions":  # Avoid overwriting n_partitions if it's special
                kwargs[param_key] = param_val

    # get the cross validation function
    cross_validate = lib_spatialize_facade.get_operator(points, kwargs["local_interpolator"],
                                                        method, kwargs["p_process"])

    if isinstance(xi, tuple):
        p_xi = deepcopy(xi)
    else:
        p_xi = xi.copy()  # Assuming xi is copyable (e.g., numpy array)

    # get the argument list
    l_args = build_arg_list(points, values, p_xi, kwargs)
    if method == "kfold":
        # Ensure correct insertion based on expected signature of cross_validate
        # The original insertion points were -2, -2.
        # This depends on the structure of l_args from build_arg_list
        # Assuming k and folding_seed are appended towards the end.
        l_args.insert(len(l_args) - 1, kwargs["folding_seed"])  # Insert before the last element (often callback)
        l_args.insert(len(l_args) - 2, k)  # Insert before seed and callback

    # run
    try:
        _, cv = cross_validate(*l_args)
    except Exception as e:
        raise SpatializeError(e)  # from e might be better for traceback

    log_message(logging.logger.info(f"using fitted model factory: {kwargs['fitted_model_factory']}"))
    return PosteriorSampleAnalyzer(cv, points, values, kwargs['fitted_model_factory'],
                                   callback=kwargs['callback'])

#todo: move this function to viz
def plot_histogram_grid_with_pdf_cdf(r, data_indices, emodels, n_rows, n_cols, bins=25, figsize=(15, 10)):
    """
    Plots a grid of histograms for specified data samples.

    For each specified sample (by index), this function visualizes its
    posterior distribution. Each subplot in the grid includes:
    - A histogram of the posterior samples for that data point.
    - The Probability Density Function (PDF) line derived from its empirical model.
    - A scaled Cumulative Distribution Function (CDF) line from its empirical model,
      overlaid for comparison.

    :param r: The full 2D array of posterior samples, where rows correspond to
              data points and columns to posterior draws.
    :type r: numpy.ndarray
    :param data_indices: A list of row indices from `r` (and corresponding keys in
                         `emodels`) for which to plot the histograms.
    :type data_indices: list[int]
    :param emodels: A dictionary or list of empirical model objects. If a dict,
                    it should be keyed by data_indices. Each emodel object must
                    have attributes `x_` (for x-axis values), `pdf_` (for PDF values),
                    and `cdf_` (for CDF values).
    :type emodels: dict[int, empirical.EmpiricalModel] or list[empirical.EmpiricalModel]
    :param n_rows: Number of rows in the subplot grid.
    :type n_rows: int
    :param n_cols: Number of columns in the subplot grid.
    :type n_cols: int
    :param bins: Number of bins to use for the histograms. Defaults to 25.
    :type bins: int, optional
    :param figsize: Overall figure size (width, height) in inches.
                    Defaults to (15, 10).
    :type figsize: tuple[float, float], optional
    """
    warm_pastel_colors = [
        "#FDBE85", "#FDD0A2", "#FCC5B0",
        "#F4A582", "#F6C5A3", "#EECFCB"
    ]

    N = len(data_indices)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten()  # Flatten to easily iterate regardless of grid shape

    for i in range(n_rows * n_cols):
        ax = axs[i]
        if i < N:
            idx = data_indices[i]

            try:
                # Access emodel, works if emodels is dict keyed by idx, or list if idx is 0-based sequential
                emodel = emodels[idx]
            except (KeyError, IndexError):
                log_message(
                    logging.logger.warning(f"Empirical model for index {idx} not found. Skipping plot for this index."))
                ax.axis('off')
                continue
            except TypeError:  # If emodels is None or not subscriptable
                log_message(logging.logger.error(
                    f"Emodels is not a valid collection (dict/list). Skipping plot for index {idx}."))
                ax.axis('off')
                continue

            # histogram the data actually fitted (post-widening, when configured) so it
            # lines up with the overlaid PDF/CDF; fall back to the raw posterior matrix
            # for models that don't carry it (e.g. built directly from a skl_model)
            data = getattr(emodel, "data_", None)
            if data is None:
                if idx >= r.shape[0]:
                    log_message(
                        logging.logger.warning(f"Index {idx} out of bounds for posterior samples matrix r. Skipping."))
                    ax.axis('off')
                    continue
                data = r[idx, :]

            color = warm_pastel_colors[i % len(warm_pastel_colors)]

            # Plot histogram
            ax.hist(data, bins=bins, density=True, histtype='stepfilled', alpha=0.3, color=color, edgecolor='black')

            # Plot PDF
            if hasattr(emodel, 'x_') and hasattr(emodel, 'pdf_'):
                ax.plot(emodel.x_, emodel.pdf_, '-k', label="PDF")
            else:
                log_message(logging.logger.warning(f"Emodel for index {idx} missing x_ or pdf_ attributes."))

            # Get current axis limits for scaling CDF
            # ymin, ymax = ax.get_ylim() # Get ylim *after* histogram and PDF are plotted for better scale

            # Plot scaled CDF
            if hasattr(emodel, 'x_') and hasattr(emodel, 'cdf_'):
                # Ensure PDF is plotted first to set a reasonable y-axis scale
                # If PDF wasn't plotted, ylim might not be representative
                current_ymin, current_ymax = ax.get_ylim()
                scaled_cdf = emodel.cdf_ * (current_ymax - current_ymin) + current_ymin
                ax.plot(emodel.x_, scaled_cdf, '-b', label="CDF (scaled)")
            else:
                log_message(
                    logging.logger.warning(f"Emodel for index {idx} missing x_ or cdf_ attributes for CDF plotting."))

            # Label
            ax.set_title(f'Sample {idx}', fontsize=10)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)  # Smaller tick labels
            ax.legend(fontsize=6)
        else:
            ax.axis('off')  # hide unused plots

    plt.tight_layout()
    plt.show()
