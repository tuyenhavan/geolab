import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.features as rfeatures
import shapely.geometry as sg
from shapely.geometry import Point


def generate_random_points_within_polygon(poly, num_points=50):
    """Generate a number of random points within polygon.

    Args:
        poly (gpd.GeoPandas): A Geopandas dataframe.
        num_points (int, optional): A number of points to generate. Defaults to 50.

    Returns:
        gpd.GeoPandas: A geopandas represents random points.
    """
    crs = poly.crs
    if crs is None:
        crs = "epsg: 4326"
    min_x, min_y, max_x, max_y = poly.union_all().bounds
    bbox = poly.union_all()
    point_list = []

    while len(point_list) < num_points:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = sg.Point([x, y])
        if bbox.contains(point):
            point_list.append(point)
    return gpd.GeoDataFrame(geometry=point_list, crs=crs)


def extract_raster_values_from_points(points, tif_path_file):
    """Extract values from a raster (.tif) using point geometries.

    Args:
        points (gpd.GeoDataFrame or list): A GeoDataFrame with point geometries
        or a list of Shapely Points.
        tif_path_file (str): Path to the raster .tif file.

    Raises:
        ValueError: If inputs are invalid or extraction fails.

    Returns:
        pd.DataFrame: A DataFrame with raster values and geometry column.
    """

    # Validate input points
    if isinstance(points, gpd.GeoDataFrame):
        point_list = points.geometry
    elif isinstance(points, list) and all(isinstance(pt, Point) for pt in points):
        point_list = gpd.GeoDataFrame(data=points, crs="epsg:4326").geometry
    else:
        raise ValueError(
            "Input must be a GeoDataFrame with points or a list of Shapely Point objects."
        )

    # Validate raster file
    if not tif_path_file.endswith(".tif"):
        raise ValueError("Please provide a valid .tif raster file path.")

    try:
        value_list = []
        with rio.open(tif_path_file) as src:
            if src.crs != point_list.crs:
                point_list = point_list.to_crs(src.crs)
            count = src.count  # Number of raster bands

            for point in point_list:
                x, y = point.x, point.y
                row, col = src.index(x, y)  # Convert to raster indices

                # Extract values from all raster bands
                pvalue = [src.read(i)[row, col] for i in range(1, count + 1)]

                # Append list if multiple bands, otherwise just value
                value_list.append(pvalue if len(pvalue) > 1 else pvalue[0])

        # Create DataFrame with raster values
        df = pd.DataFrame(value_list)
        df.columns = [f"band_{i + 1}" for i in range(df.shape[1])]
        df["geometry"] = point_list
        return df
    except ValueError as e:
        print(f"Error extracting raster values: {e}")


def raster_to_vector(tif_path, output_path=None, output_format="ESRI Shapefile"):
    """
    Converts a discrete raster file into a vector representation (polygons).

    Parameters:
    -----------
    tif_path : str
        Path to the input raster file (.tif).
    output_path : str, optional
        Path to save the output vector file (e.g., 'output.shp' or 'output.geojson').
    output_format : str, optional, default="ESRI Shapefile"
        Format to save the vector file. Supported formats include:
        - "ESRI Shapefile" (.shp)
        - "GeoJSON" (.geojson)
        - "GPKG" (.gpkg)

    Returns:
    --------
    gpd.GeoDataFrame
        A GeoDataFrame containing vectorized polygons with raster values as attributes.

    Example:
    --------
    >>> gdf = raster_to_vector("input.tif", "output.shp")
    >>> print(gdf.head())
    """

    try:
        with rio.open(tif_path) as src:
            img = src.read(1)  # Read the first raster band
            transform = src.transform
            mask = img != src.nodata  # Mask NoData values

            # Extract shapes (polygons) from raster
            polygons = [
                {"geometry": sg.shape(geom), "properties": {"value": value}}
                for geom, value in rfeatures.shapes(img, mask=mask, transform=transform)
            ]
            # Create a GeoDataFrame
            gdf = gpd.GeoDataFrame(polygons, crs=src.crs)
            gdf["properties"] = [i["value"] for i in gdf["properties"]]
            gdf = gdf.dropna()
            # Save the output file if output_path is provided
            if output_path:
                gdf.to_file(output_path, driver=output_format)

        return gdf

    except FileNotFoundError as e:
        print(f"Error processing raster file: {e}")
        return None


def generate_tiles_cover_aoi(poly, tile_size=10):
    """
    Generates a grid of square tiles that fully cover the given area of interest (AOI),
    keeping only tiles that intersect with the provided polygon.

    Parameters:
    -----------
    poly : geopandas.GeoDataFrame
        A GeoDataFrame containing the polygon(s) that define the area of interest (AOI).
    tile_size : float, optional
        The size of each tile in the same unit as the input polygon (default is 10).

    Returns:
    --------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the generated tiles (as polygons) that intersect with the AOI.

    Example:
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Polygon
    >>> poly = gpd.GeoDataFrame(geometry=[Polygon([(0,0), (10,0), (10,10), (0,10)])], crs="EPSG:4326")
    >>> tiles = generate_tiles_cover_aoi(poly, tile_size=5)
    >>> print(tiles)
    """
    xmin, ymin, xmax, ymax = poly.total_bounds
    tiles = []

    # Generate grid tiles over the bounding box
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            tile = sg.box(x, y, x + tile_size, y + tile_size)

            # Check if the tile intersects the study area
            if poly.intersects(tile).any():
                tiles.append(tile)

            y += tile_size
        x += tile_size
    # Convert to a GeoDataFrame
    result = gpd.GeoDataFrame(geometry=tiles, crs=poly.crs)
    result["codes"] = [f"A{i+1}" for i in range(len(result))]
    return result
