import numpy as np

from bootstrapx.generators.timeseries import mbb_resample


def test_mbb_hotfix_generates_non_identical_batches():
    rng = np.random.default_rng(42)
    data = np.cumsum(rng.standard_normal(200))
    gen = mbb_resample(
        data, n_resamples=8, batch_size=8, rng=np.random.default_rng(123), block_length=10
    )
    batch = next(gen)
    means = np.mean(batch, axis=1)
    assert np.std(means) > 0, f"All MBB samples appear identical: {means}"
