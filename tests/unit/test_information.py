"""Information measures (spatialize.futures.esmi).

Unit tests on this checkout's in-place build (see tests/unit/conftest.py)."""
import numpy as np

import cases
import snapshot_lib as sl

LIB = sl.load_lib()
DATA = cases.datasets()


def test_mutual_information_marginals_integrate_the_joint_density():
    """The marginal entropy read from the joint partition is that of the joint histogram density
    integrated over the other variable (computed here on a fine grid); the projections of the cells
    overlap, so treating them as disjoint bins would overstate it."""
    from spatialize.futures.esmi._main import SpatialMutualInformation
    rng = np.random.default_rng(cases.GENERATOR_SEED)
    u = rng.normal(size=300)
    v = 0.5 * u + rng.normal(size=300)
    smi = SpatialMutualInformation(T=300, M=5, alpha_m=0.8)
    _, parts, leaves = smi._calculate_joint_entropy(u, v, 1)
    for marginal, (lo, hi) in (("u", (1, 2)), ("v", (3, 4))):
        h = []
        for part, leaf in zip(parts, leaves.T):
            ids, counts = np.unique(leaf, return_counts=True)
            cells = np.asarray(part)[ids]
            x = np.linspace(cells[:, lo].min(), cells[:, hi].max(), 20001)
            mid, dx = (x[:-1] + x[1:]) / 2, x[1] - x[0]
            f = np.zeros_like(mid)
            for c, n in zip(cells, counts):
                if c[hi] > c[lo]:
                    f += ((mid >= c[lo]) & (mid < c[hi])) * (n / smi.T) / (c[hi] - c[lo])
            h.append(-np.sum(f[f > 0] * np.log2(f[f > 0])) * dx)
        assert abs(smi._calculate_marginal_entropy_from_joint(parts, leaves, marginal) - np.mean(h)) < 1e-3
