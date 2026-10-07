import libspatialize as lsp

import numpy as np

from spatialize import SpatializeError, logging, session
from spatialize._util import in_notebook
from spatialize.logging import log_message

SEP = ""

class local_interpolator:
    [IDW, KRIGING, ADAPTIVE_IDW] = ["idw", "kriging", "adaptive" + SEP + "idw"]


class partitioning_process:
    [MONDRIAN, VORONOI] = ["mondrian", "voronoi"]


PLAIN_INTERPOLATOR = "plain"
PLAINIDW = PLAIN_INTERPOLATOR + SEP + local_interpolator.IDW
[MONDRIANIDW, MONDRIANKRIGING, MONDRIANADAPTIVEIDW] = [partitioning_process.MONDRIAN + SEP + local_interpolator.IDW,
                                                       partitioning_process.MONDRIAN + SEP + local_interpolator.KRIGING,
                                                       partitioning_process.MONDRIAN + SEP + local_interpolator.ADAPTIVE_IDW
                                                       ]
VORONOIIDW = partitioning_process.VORONOI + local_interpolator.IDW


class lib_spatialize_facade:
    function_hash_map = {
        2: {MONDRIANIDW: {"estimate": lsp.estimation_esi_idw,
                          "loo": lsp.loo_esi_idw,
                          "kfold": lsp.kfold_esi_idw},
            MONDRIANKRIGING: {"estimate": lsp.estimation_esi_kriging_2d,
                              "loo": lsp.loo_esi_kriging_2d,
                              "kfold": lsp.kfold_esi_kriging_2d},
            MONDRIANADAPTIVEIDW: {"estimate": lsp.estimation_adaptive_esi_idw_2d,
                                  "loo": lsp.loo_adaptive_esi_idw_2d,
                                  "kfold": lsp.kfold_adaptive_esi_idw_2d},
            VORONOIIDW: {"estimate": lsp.estimation_voronoi_idw,
                         "loo": lsp.loo_voronoi_idw,
                         "kfold": lsp.kfold_voronoi_idw},
            PLAINIDW: {"estimate": lsp.estimation_nn_idw,
                       "loo": lsp.loo_nn_idw,
                       "kfold": lsp.kfold_nn_idw},
            },
        3: {MONDRIANIDW: {"estimate": lsp.estimation_esi_idw,
                          "loo": lsp.loo_esi_idw,
                          "kfold": lsp.kfold_esi_idw},
            MONDRIANKRIGING: {"estimate": lsp.estimation_esi_kriging_3d,
                              "loo": lsp.loo_esi_kriging_3d,
                              "kfold": lsp.kfold_esi_kriging_3d},
            MONDRIANADAPTIVEIDW: {"estimate": lsp.estimation_adaptive_esi_idw_3d,
                                  "loo": lsp.loo_adaptive_esi_idw_3d,
                                  "kfold": lsp.kfold_adaptive_esi_idw_3d},
            PLAINIDW: {"estimate": lsp.estimation_nn_idw,
                       "loo": lsp.loo_nn_idw,
                       "kfold": lsp.kfold_nn_idw},
            },
        4: {MONDRIANIDW: {"estimate": lsp.estimation_esi_idw,
                          "loo": lsp.loo_esi_idw,
                          "kfold": lsp.kfold_esi_idw},
            PLAINIDW: {"estimate": lsp.estimation_nn_idw,
                       "loo": lsp.loo_nn_idw,
                       "kfold": lsp.kfold_nn_idw},
            },
        5: {MONDRIANIDW: {"estimate": lsp.estimation_esi_idw,
                          "loo": lsp.loo_esi_idw,
                          "kfold": lsp.kfold_esi_idw},
            PLAINIDW: {"estimate": lsp.estimation_nn_idw,
                       "loo": lsp.loo_nn_idw,
                       "kfold": lsp.kfold_nn_idw},
            },
    }

    esi_kriging_models = {
        "spherical": 1,
        "exponential": 2,
        "cubic": 3,
        "gaussian": 4
    }

    custom_esi = lsp.estimation_custom_esi
    get_partitions_using_esi = lsp.get_partitions_using_esi
    get_leaf_for_samples_using_esi = lsp.get_leaf_for_samples_using_esi

    @classmethod
    def get_custom_esi_operator(cls):
        return _in_session_domain(cls.custom_esi, "estimate", queries_at=-4)

    @classmethod
    def get_operator(cls, points, local_interpolator, operation, partitioning_process):
        d = int(points.shape[1])

        if d not in lib_spatialize_facade.function_hash_map:
            raise SpatializeError(f"Points dimension must be in {list(lib_spatialize_facade.function_hash_map.keys())}")

        operator = lib_spatialize_facade.raw_operator(local_interpolator, partitioning_process)

        if operator not in lib_spatialize_facade.function_hash_map[d]:
            raise SpatializeError(f"Local interpolator '{operator}' not supported for {str(d).upper()}-D data")

        if operation not in lib_spatialize_facade.function_hash_map[d][operator]:
            raise SpatializeError(f"Operation '{operation}' not supported for '{operator}' and "
                                  f"{str(d).upper()}-D data")

        log_message(logging.logger.debug(f"esi operation: {operation}; local operator: {operator}"))
        function = lib_spatialize_facade.function_hash_map[d][operator][operation]
        if operator == PLAINIDW:  # no partitions, so no domain
            return function
        return _in_session_domain(function, operation, queries_at=-2)

    @classmethod
    def get_kriging_model_number(cls, model):
        return lib_spatialize_facade.esi_kriging_models[model]

    @classmethod
    def raw_operator(cls, local_interpolator, partitioning_process, backend=None):
        log_message(logging.logger.debug(f"partitioning process: {partitioning_process}"))

        if backend is None:  # setting the backend automatically
            if in_notebook():
                log_message(logging.logger.debug("context: in notebook"))

                # at the moment, runs IN_MEMORY, but when it's available,
                # we will have to check if there is GPU, or if we are
                # in google-colab, and run on GPU.
                return partitioning_process + local_interpolator
            else:
                # run IN_MEMORY
                log_message(logging.logger.debug("context: out of notebook"))
                return partitioning_process + local_interpolator

        raise SpatializeError(f"Backend '{backend}' not implemented for local interpolator '{local_interpolator}'")


def _in_session_domain(function, operation, queries_at):
    """Wrap a compiled entry point so that its partitions are drawn on the session domain.

    The engine draws the partitions on the box of the samples and the queries. Adding the two
    opposite corners of the domain to the queries makes that box equal to the domain; the rows of
    the corners are then removed from an estimation's output. Without a session domain the entry
    point is called unchanged.
    """
    def call(*args):
        samples = args[0]
        corners = session._domain_corners(int(np.shape(samples)[1]))
        if corners is None:
            return function(*args)
        args = list(args)
        queries = np.asarray(args[queries_at], dtype=np.float32)
        session._check_inside(corners, "data", samples)
        session._check_inside(corners, "queries", queries)
        args[queries_at] = np.vstack([queries, corners])
        estimation, members = function(*args)
        if operation == "estimate":
            members = members[:-2]
            if estimation is not None:
                estimation = estimation[:-2]
        return estimation, members

    return call
