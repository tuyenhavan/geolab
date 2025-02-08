import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
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
    if crs  is None:
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
    except Exception as e:
        raise ValueError(f"Error extracting raster values: {e}")
