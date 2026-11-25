"""Direct elongation queries and high/low smoke heatmaps."""
from spatial_isovist_core import (
    HIGH_SMOKE_VISIBILITY_M, LOW_SMOKE_VISIBILITY_M, get_engine, metric_cli,
)


def query_high_smoke(x, y, z, *, floor=None):
    """Return the metric at Unity XYZ (NaN outside domain or on stairs)."""
    return get_engine().query_high_smoke(x, y, z, floor=floor).elongation


def query_low_smoke(x, y, z, *, floor=None):
    """Return the metric at Unity XYZ (NaN outside domain or on stairs)."""
    return get_engine().query_low_smoke(x, y, z, floor=floor).elongation


if __name__ == '__main__':
    metric_cli('elongation')
