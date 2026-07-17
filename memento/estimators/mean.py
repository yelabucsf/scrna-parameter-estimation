"""Mean estimator compatibility exports.

The active implementations are kept in :mod:`memento.estimator`. Re-exporting
them here preserves the intended submodule API without maintaining a second,
divergent copy of the same numerical code.
"""

from memento.estimator import (
    _good_mean_only,
    _hyper_1d_relative,
    _mean_only_1p,
    _pseudobulk,
)

__all__ = [
    "_good_mean_only",
    "_hyper_1d_relative",
    "_mean_only_1p",
    "_pseudobulk",
]
