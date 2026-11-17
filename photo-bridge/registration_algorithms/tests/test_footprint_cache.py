
import numpy as np

from registration_algorithms.footprint_cache import (
    cache_available, read_footprint_cache, write_footprint_cache)


def test_footprint_cache_roundtrip(tmp_path):
    path = tmp_path / "samples.npz"
    samples = {
        "p0": {"visible": (np.array([1.0, 2.0]), np.array([3.0, 4.0]),
                            np.array([5.0, 6.0])),
               "thermal": (np.array([7.0]), np.array([8.0]), np.array([9.0]))},
        "p1": {"visible": (np.array([10.0]), np.array([11.0]), np.array([12.0]))},
    }
    assert not cache_available(path)
    write_footprint_cache(path, samples)
    assert cache_available(path)
    loaded = read_footprint_cache(path)
    assert set(loaded) == {"p0", "p1"}
    np.testing.assert_allclose(loaded["p0"]["visible"][0], [1.0, 2.0])
    np.testing.assert_allclose(loaded["p0"]["thermal"][2], [9.0])
    np.testing.assert_allclose(loaded["p1"]["visible"][1], [11.0])
    assert "thermal" not in loaded["p1"]
