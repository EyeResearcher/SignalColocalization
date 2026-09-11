from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from colocalize.readers import ImageReader


def test_oir_is_read_with_oirfile_and_normalized_to_czyx(monkeypatch):
    source = np.arange(2 * 4 * 3 * 5 * 6, dtype=np.uint16).reshape(2, 4, 3, 5, 6)

    class Channel:
        def __init__(self, name):
            self.name = name

    class FakeOirFile:
        dims = ("T", "Z", "C", "Y", "X")
        coords = {"C": np.asarray(["DAPI", "Signal 1", "Signal 2"])}
        channels = (Channel("DAPI"), Channel("Signal 1"), Channel("Signal 2"))

        def __init__(self, path, *, squeeze):
            assert path == Path("example.oir")
            assert squeeze is False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def asarray(self):
            return source

    monkeypatch.setitem(sys.modules, "oirfile", types.SimpleNamespace(OirFile=FakeOirFile))

    image = ImageReader(time_index=1).read_stack("example.oir")

    assert image.data.shape == (3, 4, 5, 6)
    np.testing.assert_array_equal(image.data, source[1].transpose(1, 0, 2, 3))
    assert image.channel_names == ("DAPI", "Signal 1", "Signal 2")


@pytest.mark.parametrize("filename", ["example.czi", "example.tif", "example.ome.tiff"])
def test_non_oir_images_are_read_with_bioio(monkeypatch, filename):
    source = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)

    class FakeBioImage:
        channel_names = ("DAPI", "Signal")
        dims = types.SimpleNamespace(order="TCZYX")

        def __init__(self, path):
            assert path == Path(filename)
            self.scene = None

        def set_scene(self, scene):
            self.scene = scene
            assert scene == 2

        def get_image_data(self, order, **selections):
            assert order == "CZYX"
            assert selections == {"T": 3}
            return source

    monkeypatch.setitem(
        sys.modules,
        "bioio",
        types.SimpleNamespace(BioImage=FakeBioImage),
    )

    image = ImageReader(time_index=3, scene_index=2).read_stack(filename)

    np.testing.assert_array_equal(image.data, source)
    assert image.channel_names == ("DAPI", "Signal")


@pytest.mark.parametrize("planar", [False, True])
def test_rgb_tiff_preserves_each_color_sample(tmp_path, planar):
    pytest.importorskip("bioio")
    import tifffile

    rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    path = tmp_path / "rgb.tif"
    tifffile.imwrite(
        path,
        rgb.transpose(2, 0, 1) if planar else rgb,
        photometric="rgb",
        planarconfig="separate" if planar else "contig",
    )

    stack = ImageReader().read_stack(path)
    assert stack.data.shape == (3, 1, 4, 5)
    assert stack.channel_names == ("red", "green", "blue")
    np.testing.assert_array_equal(stack.data[:, 0], rgb.transpose(2, 0, 1))
    projected = ImageReader().read(path)
    np.testing.assert_array_equal(projected.channel(2), rgb[:, :, 2])
    np.testing.assert_array_equal(projected.channel("red"), rgb[:, :, 0])


def test_oir_rgb_samples_become_channels():
    source = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(4, 5, 3)

    result = ImageReader()._to_czyx(source, ("Y", "X", "S"), Path("rgb.oir"))

    assert result.shape == (3, 1, 4, 5)
    np.testing.assert_array_equal(result[:, 0], source.transpose(2, 0, 1))


def test_oir_rejects_nonzero_scene(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "oirfile",
        types.SimpleNamespace(OirFile=object),
    )

    with pytest.raises(IndexError, match="contain one scene"):
        ImageReader(scene_index=1).read_stack("example.oir")
