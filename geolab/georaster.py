import geopandas as gpd
import rasterio as rio
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
