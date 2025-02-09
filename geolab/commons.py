import os
from collections import Counter
from datetime import datetime, timedelta

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from colour import Color
from dateutil.relativedelta import relativedelta
from matplotlib.colors import LinearSegmentedColormap


def list_files(path, ext=".tif"):
    """
    Lists all the .tif files in a given directory.

    Args:
    """
    flist = []
    for dirs, _, files in os.walk(path):
        for file in files:
            if file.endswith(f"{ext}"):
                flist.append(os.path.join(dirs, file))
    return flist


def numpy_to_xarray(numpy_array, source):
    """Convert 1-d or n-d numpy to DataArray
    args:
    numpy_array (numpy): 1-D or n-D numpy array
    source (DataArray): A xarray DataArray with original coordiates and dims

    return: A DataArray
    """
    shape = numpy_array.shape
    if len(shape) <= 1:  # If numpy data is 1-D array
        row, col = source.shape
        data = np.array(numpy_array).reshape(row, col)
        ds = xr.DataArray(data, dims=source.dims, coords=source.coords)
        return ds
    else:
        # 2-D or n-D numpy array and now it needs to assign coords
        ds = xr.DataArray(numpy_array, dims=source.dims, coords=source.coords)
        return ds


def calculate_unique_percent(ds):
    """Compute the percentage of uniques in a DataArray
    args:
    ds (DataArray): A DataArray or numpy array (2-d)

    return: A dictionary
    """
    if isinstance(ds, np.ndarray):
        data = ds.flatten()
        data = data[~np.isnan(data)]
    else:
        data = ds.values.flatten()
        data = data[~np.isnan(data)]
    # Using count the count the unique values
    count = Counter(data)
    percent = {
        str(key): round(val / sum(count.values()) * 100, 2)
        for key, val in count.items()
    }
    return percent


def cmap_generator(clist):
    """Generate a custom cmap.

    Args:
        clist (list): A list of colors.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: A custom color map output.
    """
    color_rgb = [Color(c).rgb for c in clist]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", color_rgb)
    plt.figure(figsize=(12, 3))
    plt.imshow(
        [list(np.arange(0, len(clist), 0.1))],
        interpolation="nearest",
        origin="lower",
        cmap=cmap,
    )
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return cmap


def weekly_date_list(start_year, start_month, start_day, number_of_week=52):
    """generate a list of weekly dates.

    Args:
        year (int): A year
        month (int): A month
        day (int): A day
        number_of_week (int, optional): A number of weeks to generate. Defaults to 52.

    Returns:
        list: A list of weekly datetimes.
    """
    if (
        not isinstance(start_year, int)
        or not isinstance(start_month, int)
        or not isinstance(start_day, int)
    ):
        raise ValueError(
            f"year {start_year}, month {start_month}, and day {start_day} must all be integers."
        )
    first_date = datetime(start_year, start_month, start_day)
    week_list = [first_date + i * timedelta(weeks=1) for i in range(number_of_week)]
    return week_list


def monthly_date_list(start_year, start_month, start_day, number_of_month=12):
    """Generate a list of monthly dates.

    Args:
        year (int): A initial year
        month (month): A start month.
        day (int): A start day.
        number_of_month (int, optional): A number of months to generate. Defaults to 12.

    Returns:
        lsit: A list of monthly dates.
    """
    first_date = datetime(start_year, start_month, start_day)
    month_list = [
        first_date + relativedelta(month=i) for i in range(1, number_of_month + 1)
    ]
    return month_list


def yearly_date_list(start_year, start_month, start_day, number_of_year=10):
    """Generate a list of years.

    Args:
        year (int): A start year.
        month (int): A start month.
        day (int): A start day.
        number_of_year (int, optional): A number of years to generate. Defaults to 10.

    Returns:
        list: A list of generated years.
    """
    start_date = datetime(start_year, start_month, start_day)
    year_list = [
        start_date.replace(year=start_date.year + i) for i in range(number_of_year)
    ]
    return year_list


def band_to_time_dim(col, date_list):
    """Rename time series dimension.

    Args:
        col (DataArray): A dataarray without datetime dimension.
        date_list (list): A list of dates corresponds to its dataarray length.

    Returns:
        DataArray: A DataArray with time dimension.
    """
    col = col.rename({"band": "time"})
    if len(col) == len(date_list):
        col["time"] = date_list
    return col


class CollectionReducer:
    """Define some common computation methods for ImageCollection."""

    def __init__(self) -> None:
        self.max = {"reducer": ee.Reducer.max()}
        self.min = {"reducer": ee.Reducer.min()}
        self.median = {"reducer": ee.Reducer.median()}
        self.sum = {"reducer": ee.Reducer.sum()}
        self.std = {"reducer": ee.Reducer.stdDev()}
        self.mean = {"reducer": ee.Reducer.mean()}


def compute_pairmean(ds, first_band=None, second_band=None, outband_name="tmean"):
    """Calculate the mean of pair of two bands in the ImageCollection.

    This function takes an ImageCollection containing two bands of interest and calculates
    the mean values for each image based on the specified bands. The resulting ImageCollection
    will have a new band representing the mean of the specified bands.

    Example: Calculate monthly mean temperature from monthly tmin and tmax variables in the TerraClimate.
        ds = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select(["tmmn","tmmx"])
        result = compute_pairmean(ds, first_band="tmmn", second_band="tmmx", outband_neam="tmean")

    Args:
        ds (ee.ImageCollection): ImageCollection containing two bands of interest.
        first_band (str, optional): Name of the first band of interest. If not provided,
            the function attempts to automatically determine the bands from the first image
            in the ImageCollection. Defaults to None.
        second_band (str, optional): Name of the second band of interest. If not provided,
            the function attempts to automatically determine the bands from the first image
            in the ImageCollection. Defaults to None.
        outband_name (str, optional): Name for the calculated mean band. Defaults to "tmean".

    Raises:
        TypeError: If the input 'ds' is not an ee.ImageCollection.
        ValueError: If band names are not provided and cannot be automatically determined from
            the first image in the ImageCollection.

    Returns:
        ee.ImageCollection: ImageCollection containing mean values from the specified bands for each image.
    """

    if not isinstance(ds, ee.ImageCollection):
        raise TypeError("Please provide input data as ee.ImageCollection.")
    dlist = ds.aggregate_array("system:time_start")

    if (first_band is None) & (second_band is None):
        bands = ds.first().bandNames().getInfo()
        if len(bands) == 2:
            first_band, second_band = bands
        else:
            raise ValueError("Please provide correct band names.")

    ds = ds.select([first_band, second_band])

    def pair_mean(m):
        t1 = ee.Date(m)
        t2 = t1.advance(1, "month")
        temp_ds = ds.filterDate(t1, t2).toBands().rename([first_band, second_band])
        mean = (
            temp_ds.select([first_band])
            .add(temp_ds.select([second_band]))
            .divide(2)
            .rename(outband_name)
        )
        mean = mean.set({"system:time_start": t1})
        return mean

    fcol = ee.ImageCollection.fromImages(dlist.map(pair_mean))
    return fcol


def scaling_data(ds, scale_factor=1):
    """Scaling ImageCollection or Image by a specified factor.

    Example: Scaling tmax and tmin variable in TerraClimate by a factor of 0.1
    (Please see band specification for scaling factor)
    ds = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select(["tmmn","tmmx"])
    outds = data_scale(ds, scale_factor=0.1)

    Args:
        ds (ee.Image|ee.ImageCollection): An ImageCollection or Image object
        scale_factor (int|float, optional): A scaling factor. Defaults to 1.

    Returns:
        ee.ImageCollection|ee.Image: An scaled ImageCollection or Image
    """

    if isinstance(ds, ee.ImageCollection):
        scaled_data = ds.map(
            lambda img: img.multiply(scale_factor).copyProperties(
                img, ["system:time_start"]
            )
        )
        return scaled_data
    if isinstance(ds, ee.Image):
        scaled_data = ds.multiply(scale_factor)
        return scaled_data


def geedate_converter(date_code):
    """Convert GEE datetime code into Python readable datetime.

        Example:
            date_code = 673056000000 # GEE datetime code
            out_date = geedate_converter(date_code)

    Args:
        date_code (int): The GEE datetime code.

    Returns:
        datetime.datetime: Python datetime
    """
    if not isinstance(date_code, int):
        raise TypeError("Please provide date code with integer type.")
    # Initialize the start date since GEE started date from 1970-01-01
    start_date = datetime.datetime(1970, 1, 1, 0, 0, 0)
    # Convert time code to number of hours
    hour_number = date_code / (60000 * 60)
    # Increase dates from an initial date by number of hours
    delta = datetime.timedelta(hours=hour_number)
    end_date = start_date + delta
    return end_date


def format_extracted_data(mdict):
    """A function to format the data extracted from raster by polygons FeatureCollection.

    Args:
        mdict (dict): A dictionary contains extracted data from an image or ImageCollection.

    Returns:
        pandas.DataFrame: A dataframe of extracted data with values of interest and its geometry.
    """
    mlist = []
    coords = []
    for item in mdict["features"]:
        mlist.append(item["properties"])
        coords.append(item["geometry"]["coordinates"])
    df = pd.DataFrame(mlist)
    df["coordinates"] = coords
    return df


def extract_image_by_polygon(ds, aoi, aggregate_method="mean", scale=1000):
    """Extracting raster values from an Image by ee.FeatureCollection.

    Args:
        ds (ee.Image): An image for extraction.
        aoi (ee.FeatureCollection): A FeatureCollection contains polygons.
        aggregate_method (str, optional): A method for aggregating raster values by polygon. Defaults to "mean".
        scale (int, optional): A scale for aggregation. Defaults to 1000.

    Returns:
        Dict: A dictionary contains extracted data and other properties.
    """
    if aggregate_method in ["max", "maximum"]:
        method = ee.Reducer.mean()
    elif aggregate_method in ["min", "minimum"]:
        method = ee.Reducer.min()
    elif aggregate_method in ["total", "sum"]:
        method = ee.Reducer.sum()
    elif aggregate_method in ["std"]:
        method = ee.Reducer.stdDev()
    else:
        method = ee.Reducer.mean()

    def polygon_value(feature):
        mean = ds.reduceRegion(reducer=method, geometry=feature.geometry(), scale=scale)
        return feature.set(mean)

    data = aoi.map(polygon_value).getInfo()
    return data


def bitwise_extract(img, from_bit, to_bit):
    """Extract cloud-related bits

    Args:
        img (ee.Image): The input image containing QA bands.
        from_bit (int): The starting bit.
        to_bit (int): The ending bit (inclusive).

    Returns:
        ee.Image: The output image with wanted bit extracts.
    """
    mask_size = ee.Number(to_bit).add(ee.Number(1)).subtract(from_bit)
    mask = ee.Number(1).leftShift(mask_size).subtract(1)
    out_img = img.rightShift(from_bit).bitwiseAnd(mask)
    return out_img


def cloud_mask(col, from_bit, to_bit, QA_band, threshold=1):
    """Mask out cloud-related pixels from an ImageCollection.

    Args:
        col (ee.ImageCollection): The input image collection.
        from_bit (int): The starting bit to extract bitmask.
        to_bit (int): The ending bit value to extract bitmask.
        QA_band (str): The quality assurance band, which contains cloud-related information.
        threshold (int|optional): The threshold that retains cloud-free pixels.

    Returns:
        ee.ImageCollection: Cloud masked ImageCollection.
    """

    def img_mask(img):
        qa_band = img.select(QA_band)
        bitmask_band = bitwise_extract(qa_band, from_bit, to_bit)
        mask_threshold = bitmask_band.lte(threshold)
        masked_band = img.updateMask(mask_threshold)
        return masked_band

    cloudless_col = col.map(img_mask)
    return cloudless_col


def shapefile_to_geojson(file_path):
    """Convert shapefile to geoJSON data.

    Args:
        file_path (str): The input shapefile path.

    Raises:
        FileNotFoundError: The shapefile doesn't exist.
        TypeError: Unsupported type.
    Returns:
        dict: A GeoJSON-like dictionary.
    """
    import shapefile

    if not isinstance(file_path, str):
        raise TypeError("Unsupported data type! Please provide a shapefile path.")
    if not file_path.endswith(".shp"):
        print("Invalid file extension. Please provide a valid shapefile path.")
        return None
    if not os.path.exists(file_path):
        raise FileNotFoundError("The file doesn't exist.")
    data = shapefile.Reader(file_path)
    geo_json = data.__geo_interface__
    return geo_json


def kelvin_celsius(col):
    """Convert temperature from Kelvin unit to celsius degree

    Args:
        col (ee.Image|ee.ImageCollection): The input image or collection for unit conversion.

    Returns:
        ee.Image|ee.ImageCollection: The converted output image or collection in celsius unit.
    """
    if isinstance(col, ee.Image):
        out_data = ee.Image(
            col.subtract(273.15).copyProperties(col, col.propertyNames())
        )
    elif isinstance(col, ee.ImageCollection):
        out_data = col.map(
            lambda img: img.subtract(273.15).copyProperties(img, img.propertyNames())
        )
    else:
        out_data = col
    return out_data


def arange(start, stop, step=1):
    result = []
    current = start
    while current < stop:
        result.append(current)
        current += step
    return result


def get_utm_zone(longitude, latitude):
    """
    Determines the UTM zone number and hemisphere (N/S) for a given latitude and longitude.

    Parameters:
    -----------
    longitude : float
        Longitude coordinate in decimal degrees.
    latitude : float
        Latitude coordinate in decimal degrees.

    Returns:
    --------
    str
        UTM zone with hemisphere (e.g., "34N", "35N").

    Example:
    --------
    >>> get_utm_zone(31.2357, 30.0444)  # Cairo, Egypt
    '36N'
    """
    # Calculate UTM zone number
    utm_zone = int((longitude + 180) / 6) + 1

    # Determine hemisphere
    hemisphere = "N" if latitude >= 0 else "S"

    return f"{utm_zone}{hemisphere}"
