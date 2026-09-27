"""Result objects private to ``utils`` (documented public extras, re-exported at
``utils/__init__.py``): a fit-comparison table and an outlier-detection report.
Not import-cycle-sensitive: pure dataclasses, no other stochpylib import.
"""

from dataclasses import dataclass, field


@dataclass
class FitResult:
    """Ranked distribution-fit comparison from :func:`stochpylib.utils.data.fit`."""

    best_: object
    best_name_: str
    table_: list = field(default_factory=list)
    failed_: dict = field(default_factory=dict)
    criterion: str = "aic"

    def summary(self):
        lines = [f"FitResult(criterion={self.criterion!r}, best={self.best_name_!r})"]
        header = f"  {'name':<14}{'aic':>12}{'bic':>12}{'ks_stat':>12}{'ks_p':>10}"
        lines.append(header)
        for row in self.table_:
            lines.append(
                f"  {row['name']:<14}{row.get('aic', float('nan')):>12.3f}"
                f"{row.get('bic', float('nan')):>12.3f}"
                f"{row.get('ks_stat', float('nan')):>12.4f}"
                f"{row.get('ks_pvalue', float('nan')):>10.4f}")
        if self.failed_:
            lines.append(f"  failed: {list(self.failed_)}")
        return "\n".join(lines)

    def __repr__(self):
        return f"FitResult(best={self.best_name_!r}, criterion={self.criterion!r}, " \
               f"n_candidates={len(self.table_)})"


@dataclass
class OutlierResult:
    """Outcome of :func:`stochpylib.utils.data.outlier_detection`."""

    mask: object
    scores: object
    threshold: float
    method: str
    indices: object
    n_outliers: int

    def __repr__(self):
        return f"OutlierResult(method={self.method!r}, n_outliers={self.n_outliers}, " \
               f"threshold={self.threshold:.4g})"
