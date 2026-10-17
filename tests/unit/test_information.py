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


def test_mutual_information_ensembles_share_their_partitions():
    """U and V are estimated on the same partitions, so two copies of one data set give identical
    members (before version 1.3 V used another seed, and its members were independent of U's)."""
    from spatialize.futures.esmi._main import SpatialMutualInformation
    s, v, q = DATA["2d"]
    for p_process in ("mondrian", "mondrian-legacy"):
        smi = SpatialMutualInformation(T=30, M=5, alpha_t=cases.ALPHA, alpha_m=0.6, seed=cases.SEED, p_process=p_process)
        smi.calculate_mutual_information(s, v, s, v, q)
        assert np.array_equal(smi.esi_samples_u, smi.esi_samples_v, equal_nan=True)


def test_mutual_information_is_near_zero_for_independent_variables():
    """With the permutation correction, U and V independent give a mutual information near zero, while
    V a copy of U with noise gives a clearly positive one."""
    from spatialize.futures.esmi._main import SpatialMutualInformation
    rng = np.random.default_rng(cases.GENERATOR_SEED)
    p = rng.random((300, 2))
    q = rng.random((40, 2))
    u = np.sin(6 * p[:, 0]) + rng.normal(0, 0.2, 300)
    smi = lambda bc: SpatialMutualInformation(T=200, M=50, alpha_t=0.8, alpha_m=0.8, exponent=2.0, seed=cases.SEED,
                                              bias_correction=bc)
    noise, copy = rng.normal(size=300), u + rng.normal(0, 0.2, 300)
    raw = np.nanmean(smi(False).calculate_mutual_information(p, u, p, noise, q))
    independent = np.nanmean(smi(True).calculate_mutual_information(p, u, p, noise, q))
    dependent = np.nanmean(smi(True).calculate_mutual_information(p, u, p, copy, q))
    assert abs(independent) < 0.05 and abs(independent) < raw / 3     # 0.031 against 0.115 uncorrected
    assert dependent > 3 * abs(independent)                           # 0.128
