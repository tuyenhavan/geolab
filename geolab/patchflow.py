import itertools

import rasterio as rio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject


class ImageInference:
    def __init__(self, path):
        self._read_raster(path)
    
    def __del__(self):
        self.ds.close()
        
    def _read_raster(self, path):
        self.path = path 
        self.ds = rio.open(path)
        self.meta = self.ds.meta
        self.width = self.meta["width"]
        self.height = self.meta["height"]
    def _calculate_offset(self, stride_x, stride_y):
        X = [x for x in range(0, self.width, stride_x)]
        Y = [y for y in range(0, self.height, stride_y)]
        offsets = list(itertools.product(X, Y))
        return offsets
    def _window_tranform_to_affine(self, window_transform):
        a, b, c, d, e, f, _, _, _ = window_transform
        return Affine(a, b, c, d, e, f)
    
    