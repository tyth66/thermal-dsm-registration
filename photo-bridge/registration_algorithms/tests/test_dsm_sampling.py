"""One footprint mask supplies both height summaries and point samples."""
import numpy as np
import pytest
import rasterio
from affine import Affine

from registration_algorithms.dsm_sampling import sample_footprint


def test_footprint_summary_and_samples_share_one_mask(tmp_path):
    """The height summary and include_samples come from one mask/erosion."""
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
        detail = sample_footprint(ds, corners, 1., include_samples=True)
        assert {key: value for key, value in detail.items()
                if key != "samples"} == summary
        east, north, z = detail["samples"]
        assert len(z) == summary["valid_pixels"]
        assert east.min() == 3.5 and east.max() == 9.5
        assert north.min() == 3.5 and north.max() == 9.5
        assert np.median(z) == summary["median_z"]


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


def test_pixel_center_corners_cover_exactly_the_square(tmp_path):
    """A square whose corners sit on pixel centers covers exactly those pixels.

    rasterio pixel i is centred at i + 0.5 while cv2 pixel i is centred at i;
    without the half-pixel shift the rasterized footprint grew by one pixel per
    side (25 instead of 16 here).
    """
    path = tmp_path / "centers.tif"
    with rasterio.open(path, "w", driver="GTiff", width=12, height=12, count=1,
                       dtype="float32", transform=Affine.identity()) as ds:
        ds.write(np.full((12, 12), 20.0, np.float32), 1)
    # Corners on the centers of raster pixels 2..5: 4x4 = 16 pixels.
    corners = np.array([[2.5, 2.5], [5.5, 2.5], [5.5, 5.5], [2.5, 5.5]], float)
    with rasterio.open(path) as ds:
        result = sample_footprint(ds, corners, 0.0)
    assert result["total_pixels"] == 16
    assert result["valid_pixels"] == 16


def test_outside_footprint_has_empty_samples(tmp_path):
    path=tmp_path/'dsm.tif'
    with rasterio.open(path,'w',driver='GTiff',width=4,height=4,count=1,
                       dtype='float32',transform=Affine.translation(100,100)) as ds:
        ds.write(np.ones((4,4),np.float32),1)
    with rasterio.open(path) as ds:
        result=sample_footprint(ds,np.array([[0,0],[1,0],[1,1],[0,1]]),.35,include_samples=True)
    assert result['valid_pixels']==0
    assert all(len(a)==0 for a in result['samples'])
