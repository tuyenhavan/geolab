import geopandas as gpd
import numpy as np
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from rasterio.mask import mask as mmask
from rasterio.merge import merge


def merge_raster_files(tif_list, outfile):
    """
    Merges multiple raster TIFF files into a single mosaic and saves it to an output file.

    Parameters:
    -----------
    tif_list : list of str
        List of file paths to the input raster TIFF files.
    outfile : str
        File path for the output merged raster.

    Returns:
    --------
    None
        The function writes the merged raster to the specified output file.

    Notes:
    ------
    - The function assumes all input rasters have the same CRS and data type.
    - The metadata is copied from the first raster file and updated accordingly.
    - The function automatically assigns band indices correctly in the output file.

    Example:
    --------
    >>> merge_raster_files(["file1.tif", "file2.tif"], "output_mosaic.tif")
    """
    dlist = [rio.open(file) for file in tif_list]
    mosaic, out_transform = merge(dlist)
    out_meta = dlist[0].meta.copy()
    out_meta.update(
        {
            "width": mosaic.shape[2],
            "height": mosaic.shape[1],
            "transform": out_transform,
        }
    )
    with rio.open(outfile, "w", **out_meta) as dst:
        for i in range(mosaic.shape[0]):
            dst.write(mosaic[i], i + 1)


def resampling_raster(
    path, outfile, scale_factor=0.5, resampling_method=Resampling.bilinear
):
    """Resamples a raster image by a specified scale factor and saves the output.

    Parameters:
    -----------
    path : str
        Path to the input raster file.
    outfile : str
        Path to save the resampled output raster.
    scale_factor : float, optional (default=0.5).
    If we want to convert 1m to 5m, scale factor = 1/5 or 30m to 20, scale factor = 30/20.
        Factor by which to scale the raster resolution.
        - Values > 1 increase resolution (upscaling).
        - Values < 1 decrease resolution (downscaling).
    resampling_method : rasterio.enums.Resampling, optional (default=Resampling.bilinear)
        The resampling method to use.
        Options include:
        - Resampling.nearest (fastest, best for categorical data)
        - Resampling.bilinear (good for continuous data)
        - Resampling.cubic (smoother results)
        - Resampling.average, etc.

    Returns:
    --------
    None
        Saves the resampled raster to `outfile`.

    Example:
    --------
    >>> resampling_raster("input.tif", "output_resampled.tif", scale_factor=2, resampling_method=Resampling.cubic)

    Notes:
    ------
    - The function modifies the raster's `width`, `height`, and `transform` based on the scaling.
    - It preserves the number of bands and metadata from the input raster.
    """
    with rio.open(path) as src:
        width = int(src.width * scale_factor)
        height = int(src.height * scale_factor)
        data = src.read(
            out_shape=(src.count, width, height), resampling=resampling_method
        )
        # Scale image transform
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]), (src.height / data.shape[-2])
        )
        out_meta = src.meta.copy()
        out_meta.update(
            {"width": data.shape[-1], "height": data.shape[-2], "transform": transform}
        )
        with rio.open(outfile, "w", **out_meta) as dst:
            for i in range(1, src.count + 1):
                dst.write(data[i - 1], i)


def clip_raster_by_polygon(poly, raster_file_path, outfile=None):
    """
    Clips a raster image using a given polygon and optionally saves the result.

    Args:
        poly (gpd.GeoDataFrame): A GeoDataFrame containing the polygon(s) for clipping.
        raster_file_path (str): Path to the raster file to be clipped.
        outfile (str, optional): Path to save the clipped raster. If None,
        the function returns the clipped image and metadata.

    Returns:
        tuple: (metadata, clipped image array) if outfile is None.
        None: If the clipped raster is saved to a file.

    Raises:
        ValueError: If input types are incorrect
        or if the polygon has no valid geometry.
        FileNotFoundError: If the raster file path does not exist.
    """

    # Validate input types
    if not isinstance(poly, gpd.GeoDataFrame):
        raise ValueError("Expected 'poly' to be a GeoDataFrame.")

    if not isinstance(raster_file_path, str):
        raise ValueError("Expected 'raster_file_path' to be a file path (string).")

    # Open raster file
    with rio.open(raster_file_path) as src:
        if src.count == 0:
            raise ValueError("Raster contains no bands.")

        # Ensure the polygon has the correct CRS
        if poly.crs != src.crs:
            poly = poly.to_crs(src.crs)

        # Extract geometries
        geoms = poly.geometry.values
        if len(geoms) == 0:
            raise ValueError("Polygon has no valid geometry.")

        # Perform masking (clipping)
        out_img, out_transform = mmask(src, geoms, crop=True)

        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "width": out_img.shape[2],
                "height": out_img.shape[1],
                "transform": out_transform,
            }
        )

    # Save clipped raster if an output file is specified
    if outfile:
        with rio.open(outfile, "w", **out_meta) as dst:
            # Write each band separately
            for i in range(out_img.shape[0]):
                dst.write(out_img[i], i + 1)
        return None  # Indicates successful save

    return out_meta, out_img


def downscale_raster(file_path, decimation=10, mask_value=None, resampling=None):
    """
    Reads a raster file, downscales it by a given factor, and converts it into an xarray DataArray
    while preserving spatial metadata.

    This function reduces the resolution of the raster by the specified decimation factor,
    applies optional masking, and returns the result as an `xarray.DataArray`. The downscaled
    raster retains geospatial metadata, including coordinate reference system (CRS) and transformation.

    Parameters:
    -----------
    file_path : str
        Path to the input raster file (GeoTIFF or other raster format supported by Rasterio).
    decimation : int, optional (default=10)
        Factor by which to reduce the resolution. A value of 10 means reducing both width and
        height by a factor of 10 (i.e., only every 10th pixel is kept).
    mask_value : int or float, optional (default=None)
        If specified, replaces all occurrences of this value in the raster with NaN.
        Useful for handling missing data or no-data values.
    resampling : rasterio.enums.Resampling, optional (default=rasterio.enums.Resampling.average)
        The resampling method used during downscaling. Common options include:
        - `Resampling.nearest` (fastest, preserves original values)
        - `Resampling.bilinear` (linear interpolation)
        - `Resampling.cubic` (smoother interpolation)
        - `Resampling.average` (default, computes the average of the downsampled area)

    Returns:
    --------
    xarray.DataArray
        An `xarray.DataArray` containing the downsampled raster with:
        - Dimensions: (`y`, `x`)
        - Coordinates: (`y`: latitude/vertical axis, `x`: longitude/horizontal axis)
        - Spatial metadata: CRS and transform stored using `rioxarray`.

    Example Usage:
    --------------
    >>> downsampled_raster = downscale_raster("example.tif", decimation=5, mask_value=0, resampling=rio.enums.Resampling.nearest)
    >>> print(downsampled_raster)

    Notes:
    ------
    - The function uses Rasterio's `read(out_shape=...)` to efficiently resample the raster during reading.
    - The default resampling method (`average`) is ideal for continuous data like elevation, but for categorical data (e.g., land cover), use `nearest`.
    - The resulting `DataArray` can be used for further geospatial analysis, visualization, or export.
    """

    # Set default resampling method if none is provided
    if resampling is None:
        resampling = Resampling.average

    # Open the raster file
    with rio.open(file_path) as src:
        # Compute new downsampled shape
        out_shape = (int(src.height / decimation), int(src.width / decimation))

        # Read and downscale the raster
        data = src.read(out_shape=out_shape, resampling=resampling).squeeze()

        # Generate new coordinate grid
        x_coords = np.linspace(src.bounds.left, src.bounds.right, out_shape[1])
        y_coords = np.linspace(src.bounds.top, src.bounds.bottom, out_shape[0])

        # Preserve CRS and transformation information
        crs = src.crs
        transform = rio.transform.from_bounds(
            *src.bounds, width=out_shape[1], height=out_shape[0]
        )

    # Apply masking (replace mask_value with NaN)
    if mask_value is not None:
        data = np.where(data == mask_value, np.nan, data)

    # Convert to xarray DataArray
    raster_array = xr.DataArray(
        data=data, dims=["y", "x"], coords={"y": y_coords, "x": x_coords}
    )

    # Attach spatial metadata
    raster_array = raster_array.rio.write_crs(crs)
    raster_array = raster_array.rio.write_transform(transform)

    return raster_array
