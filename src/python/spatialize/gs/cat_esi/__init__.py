from ._main import (
    cat_esi_griddata,
    cat_esi_nongriddata,
    cat_esi_hparams_search,
    CatESIResult,
    CatESIGridSearchResult,
)
from .agg_functions import (
    aggregate_with_mv,
    aggregate_with_ordinal_mv,
    categorical_feature_precision,
    categorical_precision_cube,
)
from .score_functions import (
    accuracy,
    f1_macro,
    f1_weighted,
    f1_micro,
    precision_macro,
    recall_macro,
    cohen_kappa,
    SCORING_OPTIONS,
)

__all__ = [
    "cat_esi_griddata",
    "cat_esi_nongriddata",
    "cat_esi_hparams_search",
    "CatESIResult",
    "CatESIGridSearchResult",
    "aggregate_with_mv",
    "aggregate_with_ordinal_mv",
    "categorical_feature_precision",
    "categorical_precision_cube",
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "f1_micro",
    "precision_macro",
    "recall_macro",
    "cohen_kappa",
    "SCORING_OPTIONS",
]

# Optional BTD functions (require compiled dawid_skene_em_cpp module)
try:
    from .agg_functions import (
        aggregate_with_btd,
        objective_function_for_spatial_penalty,
        optimize_btd_spatial_penalty,
    )
    __all__ += [
        "aggregate_with_btd",
        "objective_function_for_spatial_penalty",
        "optimize_btd_spatial_penalty",
    ]
except ImportError:
    pass
