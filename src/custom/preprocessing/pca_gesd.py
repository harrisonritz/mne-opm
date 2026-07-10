"""Generic PCA-whitened GESD outlier detection.

This module factors out the PCA→GESD procedure shared by two analyses:

* ICA component selection (:mod:`custom.preprocessing.auto_ica`), where the
  *items* are independent components, and
* bad-channel detection (:mod:`custom.preprocessing.bad_channels`), where the
  *items* are sensor channels.

Given a set of named per-item *metrics* (each with an outlier direction), the
procedure:

1. assembles a metric matrix (``k`` metrics × ``n`` items),
2. z-scores each metric across items,
3. reduces the metrics to orthogonal principal components (PCA via SVD),
   yielding one *eigenscore* per item per component,
4. runs a generalized ESD (GESD) test on each eigenscore, with the family-wise
   error rate controlled by a Šidák correction across the retained components,
5. returns the union of items flagged on any component.

Folding several heterogeneous detectors into one PCA→GESD family removes the
redundancy and uncontrolled error rate of running independent thresholded
detectors, while keeping each metric individually selectable.

A small set of normality-improving transforms (:func:`log_transform`,
:func:`signed_sqrt`, :func:`fisher_z`, :func:`logit`) is provided so callers can
pre-condition skewed/bounded metrics before they are z-scored.

Author: Harrison Ritz, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from osl_ephys.preprocessing.osl_wrappers import gesd as osl_gesd


# ---------------------------------------------------------------------------
# Normality-improving transforms
# ---------------------------------------------------------------------------


def log_transform(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Natural log of a positive, right-skewed metric (``log(x + eps)``)."""
    return np.log(np.asarray(x, dtype=float) + eps)


def signed_sqrt(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Signed square root — reduces skew while preserving sign (e.g. kurtosis)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.sqrt(np.abs(x) + eps)


def fisher_z(x: np.ndarray, clip: float = 0.999) -> np.ndarray:
    """Fisher z-transform (``arctanh``) for correlation-like metrics in [-1, 1].

    Values are clipped to ``[-clip, clip]`` first so the transform stays finite.
    """
    x = np.clip(np.asarray(x, dtype=float), -clip, clip)
    return np.arctanh(x)


def logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Logit transform for proportions in [0, 1] (``log(x / (1 - x))``).

    Values are clipped to ``[eps, 1 - eps]`` so the transform stays finite.
    """
    x = np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MetricSpec:
    """A single per-item metric fed into the PCA→GESD procedure.

    Parameters
    ----------
    name : str
        Identifier for the metric (e.g. ``"log_std"``, ``"eog"``).
    values : np.ndarray
        Per-item metric values, shape ``(n_items,)``. Should already carry any
        normality-improving transform; the procedure only z-scores them.
    side : int
        Outlier direction for GESD: ``1`` = high values are bad, ``-1`` = low
        values are bad, ``0`` = both tails.
    """

    name: str
    values: np.ndarray
    side: int


@dataclass
class PCAGesdResult:
    """Outputs of :func:`run_pca_gesd`, consumed by TSV writers and figures.

    Attributes
    ----------
    metric_names : list of str
        Names of the input metrics, in row order of ``M``.
    sides : np.ndarray
        Outlier side per input metric, shape ``(k,)``.
    M : np.ndarray
        Raw (pre-standardization) metric matrix, shape ``(k, n_items)``.
    M_std : np.ndarray
        Standardized metric matrix, shape ``(k, n_items)``.
    loadings : np.ndarray
        PC loadings (how each metric weights each PC), shape ``(k, n_pcs)``.
    eigenscores : np.ndarray
        PC scores (how each item loads on each PC), shape ``(n_pcs, n_items)``.
    var_explained : np.ndarray
        Variance fraction explained by each retained PC, shape ``(n_pcs,)``.
    var_explained_all : np.ndarray
        Variance fraction for the full singular spectrum (for the scree plot).
    n_pcs : int
        Number of retained principal components.
    pc_sides : np.ndarray
        Outlier side used for GESD on each PC, shape ``(n_pcs,)``.
    alpha : float
        Overall family-wise significance level.
    alpha_per_pc : float
        Šidák-corrected per-PC significance level.
    per_pc_flagged : list of np.ndarray
        Boolean mask of flagged items for each PC, each shape ``(n_items,)``.
    flagged : np.ndarray
        Union boolean mask of flagged items, shape ``(n_items,)``.
    n_items : int
        Number of items.
    """

    metric_names: List[str]
    sides: np.ndarray
    M: np.ndarray
    M_std: np.ndarray
    loadings: np.ndarray
    eigenscores: np.ndarray
    var_explained: np.ndarray
    var_explained_all: np.ndarray
    n_pcs: int
    pc_sides: np.ndarray
    alpha: float
    alpha_per_pc: float
    per_pc_flagged: List[np.ndarray]
    flagged: np.ndarray
    n_items: int


# ---------------------------------------------------------------------------
# Core procedure
# ---------------------------------------------------------------------------


def sanitize_values(values: np.ndarray) -> np.ndarray:
    """Replace non-finite entries with the finite median (or 0 if all bad)."""
    vals = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(vals)
    if not finite.all():
        fill = float(np.median(vals[finite])) if finite.any() else 0.0
        vals[~finite] = fill
    return vals


def clean_specs(
    specs: Sequence[MetricSpec],
    n_items: int,
    log: Optional[Callable[[str], None]] = None,
) -> List[MetricSpec]:
    """Sanitize NaNs and drop wrong-length / zero-variance metrics."""
    _log = log or (lambda _m: None)
    out: List[MetricSpec] = []
    for s in specs:
        vals = sanitize_values(s.values)
        if vals.size != n_items:
            _log(f"metric '{s.name}' length {vals.size} != {n_items}; dropping")
            continue
        if np.std(vals) == 0:
            _log(f"metric '{s.name}' is constant; dropping")
            continue
        out.append(MetricSpec(s.name, vals, s.side))
    return out


def sidak_alpha(alpha: float, n_tests: int) -> float:
    """Šidák-corrected per-test significance level for ``n_tests`` tests."""
    return 1.0 - (1.0 - alpha) ** (1.0 / max(1, n_tests))


def gesd_flag_eigenscores(
    eigenscores: np.ndarray,
    pc_sides: np.ndarray,
    alpha_per_pc: float,
    p_out: float,
) -> tuple[List[np.ndarray], np.ndarray]:
    """Run GESD on each eigenscore (PC) and union the flagged items.

    Parameters
    ----------
    eigenscores : np.ndarray
        ``(n_pcs, n_items)`` PC scores.
    pc_sides : np.ndarray
        Outlier side per PC (``1``/``-1``/``0``).
    alpha_per_pc : float
        Per-PC significance level (already Šidák-corrected).
    p_out : float
        Maximum fraction of outliers per PC passed to GESD.

    Returns
    -------
    per_pc_flagged : list of np.ndarray
        Boolean mask per PC.
    flagged : np.ndarray
        Union boolean mask.
    """
    n_items = eigenscores.shape[1]
    flagged = np.zeros(n_items, dtype=bool)
    per_pc: List[np.ndarray] = []
    for p in range(eigenscores.shape[0]):
        side = int(pc_sides[p])
        flags, _ = osl_gesd(
            eigenscores[p, :], alpha=alpha_per_pc, p_out=p_out, outlier_side=side
        )
        flags = np.asarray(flags, dtype=bool)
        per_pc.append(flags)
        flagged |= flags
    return per_pc, flagged


def empty_result(
    metric_names: Sequence[str],
    sides: Sequence[int],
    n_items: int,
    alpha: float = 0.05,
) -> PCAGesdResult:
    """Build a no-op result (no PCs, nothing flagged) for ``n_items`` items."""
    k = len(list(metric_names))
    return PCAGesdResult(
        metric_names=list(metric_names),
        sides=np.asarray(sides, dtype=float),
        M=np.empty((k, n_items)),
        M_std=np.empty((k, n_items)),
        loadings=np.empty((k, 0)),
        eigenscores=np.empty((0, n_items)),
        var_explained=np.empty(0),
        var_explained_all=np.empty(0),
        n_pcs=0,
        pc_sides=np.empty(0),
        alpha=alpha,
        alpha_per_pc=alpha,
        per_pc_flagged=[],
        flagged=np.zeros(n_items, dtype=bool),
        n_items=n_items,
    )


def run_pca_gesd(
    specs: Sequence[MetricSpec],
    *,
    alpha: float = 0.05,
    p_out: float = 1.0,
    var_threshold: float = 0.99,
    n_pcs: Optional[int] = None,
    min_items: int = 5,
    log: Optional[Callable[[str], None]] = None,
) -> PCAGesdResult:
    """Run the PCA-whitened GESD procedure over a set of per-item metrics.

    Parameters
    ----------
    specs : sequence of MetricSpec
        Per-item metrics (already transformed). All must share ``n_items``.
    alpha : float
        Overall family-wise significance level (default 0.05).
    p_out : float
        Maximum fraction of outliers per PC passed to GESD (default 1.0).
    var_threshold : float
        Cumulative variance fraction used to choose the number of PCs when
        ``n_pcs`` is None (default 0.99).
    n_pcs : int | None
        Fixed number of PCs to retain; overrides ``var_threshold`` when given.
    min_items : int
        Minimum number of items required to run; otherwise a no-op result is
        returned (default 5).
    log : callable | None
        Optional logging callback.

    Returns
    -------
    result : PCAGesdResult
    """
    _log = log or (lambda _m: None)

    specs = list(specs)
    names0 = [s.name for s in specs]
    sides0 = [s.side for s in specs]
    if not specs:
        _log("No metrics provided to PCA-GESD; nothing to do.")
        return empty_result(names0, sides0, 0, alpha)

    n_items = int(np.asarray(specs[0].values).reshape(-1).shape[0])
    specs = clean_specs(specs, n_items, log)
    names = [s.name for s in specs]
    sides = np.array([s.side for s in specs], dtype=float)

    if not specs:
        _log("No usable metrics after cleaning; nothing to do.")
        return empty_result(names, sides, n_items, alpha)

    if n_items < min_items:
        _log(f"Too few items ({n_items} < {min_items}); skipping PCA-GESD.")
        return empty_result(names, sides, n_items, alpha)

    # Metric matrix: k metrics x n items.
    M = np.vstack([s.values for s in specs])
    k = M.shape[0]
    _log(f"Metric matrix: {k} metrics x {n_items} items")
    _log("=== Metric summary ===")
    for i, s in enumerate(specs):
        direction = {1: "high=bad", -1: "low=bad", 0: "both tails"}[s.side]
        _log(
            f"  {s.name}: mean={np.mean(M[i]):.3f}, std={np.std(M[i]):.3f} "
            f"({direction})"
        )

    # Standardize each metric (row-wise).
    M_std = StandardScaler().fit_transform(M.T).T  # (k, n_items)

    # PCA via SVD of the standardized matrix.
    U, sv, _ = np.linalg.svd(M_std, full_matrices=False)
    var_explained_all = (sv**2) / (sv**2).sum()

    if n_pcs is None:
        cumvar = np.cumsum(var_explained_all)
        n_pcs = int(np.searchsorted(cumvar, var_threshold) + 1)
    n_pcs = max(1, min(n_pcs, k))

    loadings = U[:, :n_pcs]  # (k, n_pcs)
    eigenscores = loadings.T @ M_std  # (n_pcs, n_items)
    var_explained = var_explained_all[:n_pcs]

    _log("=== PCA variance explained ===")
    for p in range(n_pcs):
        _log(f"  PC{p + 1}: {var_explained[p] * 100:.1f}%")
    _log(f"  Total ({n_pcs} PCs): {var_explained.sum() * 100:.1f}%")
    _log("=== PC loadings (metric weights) ===")
    for p in range(n_pcs):
        parts = [f"{names[i]}={loadings[i, p]:.2f}" for i in range(k)]
        _log(f"  PC{p + 1}: {', '.join(parts)}")

    # dot product approach
    # pc_sides = np.sign(loadings.T @ sides)

    # Per-PC tail direction: product of sign(loading * side) across metrics.
    # If any metric has side=0 (both tails), the product is 0 → test both tails.
    # pc_sides = np.prod(np.sign(loadings * sides[:, np.newaxis]), axis=0)

    # Voting approach: sum of absolute loadings per side.
    # print("calculating vote per side")
    # vote = {s: np.sum(np.abs(loadings[sides == s, :]), axis=0) for s in [-1, 0, 1]}
    # _log(f"Vote per side: {vote}")
    # print("get pc_sides")
    # pc_sides = max(vote, key=lambda s: vote[s])  # vectorised with np.argmax
    # print(f"pc_sides: {pc_sides}")

    # Per-PC tail direction: weighted plurality vote across metrics.
    # For each PC, sum |loading| separately for side={-1, 0, 1}; the winning
    # side becomes pc_side.  side=0 (both-tails) only wins if two-tailed metrics
    # collectively carry more loading weight on that PC than either directed group.
    vote_matrix = np.stack(
        [np.abs(np.sum(loadings[sides == s, :], axis=0)) for s in [-1, 0, 1]],
        axis=0,
    )  # (3, n_pcs); rows correspond to sides [-1, 0, 1]
    pc_sides = np.array([-1, 0, 1])[np.argmax(vote_matrix, axis=0)]
    _log(f"PC sides (vote): {pc_sides.tolist()}")

    # Šidák correction across PCs, then GESD per eigenscore.
    alpha_per_pc = sidak_alpha(alpha, n_pcs)
    _log(f"Šidák-corrected alpha: {alpha:.3f} -> {alpha_per_pc:.4f} per PC")

    per_pc_flagged, flagged = gesd_flag_eigenscores(
        eigenscores, pc_sides, alpha_per_pc, p_out
    )
    _log("=== GESD results per PC ===")
    for p in range(n_pcs):
        side_str = {1: "upper", -1: "lower", 0: "both"}.get(int(pc_sides[p]), "both")
        n_flag = int(per_pc_flagged[p].sum())
        idx = np.where(per_pc_flagged[p])[0].tolist()
        if n_flag:
            _log(f"  PC{p + 1} ({side_str}): {n_flag} outliers -> {idx}")
        else:
            _log(f"  PC{p + 1} ({side_str}): no outliers")
    _log(f"Total flagged items: {int(flagged.sum())}")

    return PCAGesdResult(
        metric_names=names,
        sides=sides,
        M=M,
        M_std=M_std,
        loadings=loadings,
        eigenscores=eigenscores,
        var_explained=var_explained,
        var_explained_all=var_explained_all,
        n_pcs=n_pcs,
        pc_sides=pc_sides,
        alpha=alpha,
        alpha_per_pc=alpha_per_pc,
        per_pc_flagged=per_pc_flagged,
        flagged=flagged,
        n_items=n_items,
    )


# ---------------------------------------------------------------------------
# Diagnostic figures
# ---------------------------------------------------------------------------


def _fig_loadings(result: PCAGesdResult):
    """Heatmap of PC loadings (metrics x PCs)."""
    import matplotlib.pyplot as plt

    k, n_pcs = result.loadings.shape
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * n_pcs + 2), max(3, 0.35 * k + 1)))
    vmax = float(np.abs(result.loadings).max()) or 1.0
    im = ax.imshow(result.loadings, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_pcs))
    ax.set_xticklabels([f"PC{p + 1}" for p in range(n_pcs)])
    ax.set_yticks(range(k))
    ax.set_yticklabels(result.metric_names, fontsize=8)
    ax.set_xlabel("Principal component")
    ax.set_title("Metric loadings on PCs")
    fig.colorbar(im, ax=ax, shrink=0.8, label="loading")
    fig.tight_layout()
    return fig


def _fig_item_eigenscores(
    result: PCAGesdResult, item_label: str, item_names: Optional[Sequence[str]]
):
    """Heatmap of item eigenscores (items x PCs); flagged items labelled red."""
    import matplotlib.pyplot as plt

    data = result.eigenscores.T  # (n_items, n_pcs)
    n_items, n_pcs = data.shape
    fig, ax = plt.subplots(
        figsize=(max(4, 0.6 * n_pcs + 2), min(20, max(3, 0.18 * n_items + 1)))
    )
    vmax = float(np.abs(data).max()) or 1.0
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_pcs))
    ax.set_xticklabels([f"PC{p + 1}" for p in range(n_pcs)])
    ax.set_ylabel(f"{item_label} index")
    ax.set_xlabel("Principal component")
    ax.set_title(f"{item_label} eigenscores (flagged in red)")
    flagged_idx = np.where(result.flagged)[0]
    if flagged_idx.size:
        if item_names is not None:
            labels = [str(item_names[i]) for i in flagged_idx]
        else:
            labels = [f"{item_label}{i}" for i in flagged_idx]
        ax.set_yticks(flagged_idx)
        ax.set_yticklabels(labels, fontsize=7, color="red")
    fig.colorbar(im, ax=ax, shrink=0.8, label="eigenscore")
    fig.tight_layout()
    return fig


def _fig_scree(result: PCAGesdResult):
    """Scree plot: variance explained per PC and cumulative."""
    import matplotlib.pyplot as plt

    ve = result.var_explained_all
    x = np.arange(1, len(ve) + 1)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(x, ve * 100, "o-", color="steelblue", label="per PC")
    ax.plot(x, np.cumsum(ve) * 100, "s--", color="gray", label="cumulative")
    ax.axvline(
        result.n_pcs + 0.5, color="red", ls=":", label=f"kept {result.n_pcs} PCs"
    )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("PCA scree")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _fig_metric_corr(result: PCAGesdResult):
    """Correlation matrix of the standardized metrics (redundancy check)."""
    import matplotlib.pyplot as plt

    if result.M_std.shape[0] < 2:
        return None
    corr = np.corrcoef(result.M_std)
    k = corr.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, 0.4 * k + 2), max(4, 0.4 * k + 2)))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(k))
    ax.set_xticklabels(result.metric_names, rotation=90, fontsize=7)
    ax.set_yticks(range(k))
    ax.set_yticklabels(result.metric_names, fontsize=7)
    ax.set_title("Metric correlation (standardized)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="r")
    fig.tight_layout()
    return fig


def _fig_standardized(result: PCAGesdResult, item_label: str):
    """Heatmap of the standardized metrics (metrics x items)."""
    import matplotlib.pyplot as plt

    data = result.M_std  # (k, n_items)
    k, n_items = data.shape
    fig, ax = plt.subplots(
        figsize=(min(25, max(5, 0.18 * n_items + 2)), max(3, 0.35 * k + 1))
    )
    vmax = float(np.abs(data).max()) or 1.0
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(k))
    ax.set_yticklabels(result.metric_names, fontsize=8)
    ax.set_xlabel(f"{item_label} index")
    ax.set_title("Standardized metrics")
    fig.colorbar(im, ax=ax, shrink=0.8, label="z")
    fig.tight_layout()
    return fig


def _fig_pc_outliers(result: PCAGesdResult, item_label: str):
    """Per-PC scatter of eigenscore vs item index; flagged items in red."""
    import matplotlib.pyplot as plt

    n_pcs = result.n_pcs
    ncol = min(4, n_pcs)
    nrow = int(np.ceil(n_pcs / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    n_items = result.eigenscores.shape[1]
    x = np.arange(n_items)
    for p in range(n_pcs):
        ax = axes[p // ncol][p % ncol]
        y = result.eigenscores[p]
        flags = result.per_pc_flagged[p]
        ax.scatter(x[~flags], y[~flags], s=12, color="steelblue", label="kept")
        if flags.any():
            ax.scatter(x[flags], y[flags], s=24, color="red", label="flagged")
        ax.set_title(f"PC{p + 1} (α/PC={result.alpha_per_pc:.4f})", fontsize=9)
        ax.set_xlabel(f"{item_label} index")
        ax.set_ylabel("eigenscore")
        ax.legend(fontsize=7)
    for j in range(n_pcs, nrow * ncol):
        fig.delaxes(axes[j // ncol][j % ncol])
    fig.suptitle("Per-PC GESD outliers")
    fig.tight_layout()
    return fig


def _fig_pc_outliers_histogram(result: PCAGesdResult, item_label: str):
    """Per-PC overlaid histograms of kept vs flagged eigenscores."""
    import matplotlib.pyplot as plt

    n_pcs = result.n_pcs
    ncol = min(4, n_pcs)
    nrow = int(np.ceil(n_pcs / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for p in range(n_pcs):
        ax = axes[p // ncol][p % ncol]
        y = result.eigenscores[p]
        flags = result.per_pc_flagged[p]
        # Shared bin edges so flagged and kept bars align.
        n_bins = max(10, int(np.sqrt(len(y))))
        bins = np.linspace(y.min(), y.max(), n_bins + 1)
        ax.hist(y[~flags], bins=bins, color="steelblue", alpha=0.7, label="kept")
        if flags.any():
            ax.hist(y[flags], bins=bins, color="red", alpha=0.7, label="flagged")
        ax.set_title(f"PC{p + 1} (α/PC={result.alpha_per_pc:.4f})", fontsize=9)
        ax.set_xlabel("eigenscore")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
    for j in range(n_pcs, nrow * ncol):
        fig.delaxes(axes[j // ncol][j % ncol])
    fig.suptitle("Per-PC GESD outliers (histogram)")
    fig.tight_layout()
    return fig


def save_pca_gesd_figures(
    result: PCAGesdResult,
    out_dir: Path,
    basename: str,
    *,
    item_label: str = "item",
    item_names: Optional[Sequence[str]] = None,
    prefix: str = "gesd",
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """Save the PCA-GESD diagnostic figures into ``out_dir``.

    Figures (each failure-isolated so plotting never aborts the analysis):
    ``{prefix}Loadings`` (metric→PC loadings), ``{prefix}Eigenscores``
    (item→PC eigenscores), ``{prefix}Scree``, ``{prefix}MetricCorr``,
    ``{prefix}StdScores`` and ``{prefix}Outliers``.

    Parameters
    ----------
    result : PCAGesdResult
        Output of :func:`run_pca_gesd`.
    out_dir : pathlib.Path
        Directory the figures are written to (created if needed).
    basename : str
        Filename prefix; figures are ``{basename}_{prefix}<Name>.png``.
    item_label : str
        Axis label for items (e.g. ``"IC"`` or ``"channel"``).
    item_names : sequence of str | None
        Item names used to label flagged items on the eigenscore heatmap.
    prefix : str
        Figure-name prefix (default ``"gesd"``).
    log : callable | None
        Optional logging callback.
    """
    import matplotlib.pyplot as plt

    _log = log or (lambda _m: None)
    if result is None or result.n_pcs == 0:
        _log("No PCs in result; skipping diagnostic figures.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "Loadings": lambda: _fig_loadings(result),
        "Eigenscores": lambda: _fig_item_eigenscores(result, item_label, item_names),
        "Scree": lambda: _fig_scree(result),
        "MetricCorr": lambda: _fig_metric_corr(result),
        "StdScores": lambda: _fig_standardized(result, item_label),
        "Outliers": lambda: _fig_pc_outliers(result, item_label),
        "OutliersHist": lambda: _fig_pc_outliers_histogram(result, item_label),
    }

    for name, builder in builders.items():
        fname = out_dir / f"{basename}_{prefix}{name}.png"
        try:
            fig = builder()
        except Exception as exc:  # figures must never break the analysis
            _log(f"Figure '{prefix}{name}' failed: {type(exc).__name__}: {exc}")
            continue
        if fig is None:
            continue
        try:
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            _log(f"Saved figure: {fname.name}")
        finally:
            plt.close(fig)
