"""One footprint mask supplies both height summaries and plane-fit samples."""
import numpy as np
import pytest
import rasterio
from affine import Affine

from registration_algorithms.dsm_sampling import sample_footprint
from registration_algorithms.application import geometry as geometry_module
from registration_algorithms.panel_thermal_model import sample_polygon_pixels


def _geometry_sampler():
    return geometry_module._samples


def test_erosion_radius_and_mask_are_identical_across_stages(tmp_path):
    path = tmp_path / "dsm.tif"
    values = np.full((16, 16), 20, np.float32)
    values[4, 4] = np.nan
    values[6, 6] = 0  # Declared nodata is inside the otherwise accepted height range.
    values[7, 7] = 80  # Exclusive elevation limit.
    mask = np.full((16, 16), 255, np.uint8)
    mask[5, 5] = 0
    with rasterio.open(path, "w", driver="GTiff", width=16, height=16, count=1,
                       dtype="float32", transform=Affine.identity(), nodata=0) as ds:
        ds.write(values, 1)
        ds.write_mask(mask)
    corners = np.array([[2,2], [10,2], [10,10], [2,10]], float)
    with rasterio.open(path) as ds:
        summary = sample_footprint(ds, corners, 1.)
        assert summary["total_pixels"] == 49
        assert summary["valid_pixels"] == 45
        for sampler in (sample_polygon_pixels, _geometry_sampler()):
            east, north, z = sampler(ds, corners, 1., minimum_pixels=1)
            assert len(z) == summary["valid_pixels"]
            assert east.min() == 3.5 and east.max() == 9.5
            assert north.min() == 3.5 and north.max() == 9.5
            assert np.median(z) == summary["median_z"]
            assert sampler(ds, corners, 1., minimum_pixels=46) is None


@pytest.mark.parametrize("erode_m", [0., .35, 5.])
def test_sample_coordinates_share_summary_for_rotated_anisotropic_raster(tmp_path, erode_m):
    path = tmp_path / "rotated.tif"
    transform = Affine.translation(596000,3234000) * Affine.rotation(17) * Affine.scale(.1,-.2)
    values = 20 + np.arange(256, dtype=np.float32).reshape(16,16) * .001
    with rasterio.open(path, "w", driver="GTiff", width=16,height=16,count=1,
                       dtype="float32",transform=transform) as ds:
        ds.write(values,1)
    corners = np.array([transform @ p for p in [(0,0),(12,0),(12,12),(0,12)]])
    with rasterio.open(path) as ds:
        summary = sample_footprint(ds,corners,erode_m)
        detail = sample_footprint(ds,corners,erode_m,include_samples=True)
        assert {k:v for k,v in detail.items() if k!='samples'} == summary
        east,north,z=detail['samples']
        assert len(z)==summary['valid_pixels']
        if not len(z):
            assert summary['median_z'] is None
            return
        col,row=(~transform) @ (east,north)
        np.testing.assert_allclose(col % 1,.5,atol=1e-7)
        np.testing.assert_allclose(row % 1,.5,atol=1e-7)
        np.testing.assert_array_equal(z,values[np.floor(row).astype(int),np.floor(col).astype(int)])
        assert np.median(z)==summary['median_z']


def test_outside_footprint_has_empty_samples(tmp_path):
    path=tmp_path/'dsm.tif'
    with rasterio.open(path,'w',driver='GTiff',width=4,height=4,count=1,
                       dtype='float32',transform=Affine.translation(100,100)) as ds:
        ds.write(np.ones((4,4),np.float32),1)
    with rasterio.open(path) as ds:
        result=sample_footprint(ds,np.array([[0,0],[1,0],[1,1],[0,1]]),.35,include_samples=True)
    assert result['valid_pixels']==0
    assert all(len(a)==0 for a in result['samples'])
