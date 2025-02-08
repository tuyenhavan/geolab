import os
from collections import Counter
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
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

def weekly_date_list(year, month, day, number_of_week=52):
    """generate a list of weekly dates.

    Args:
        year (int): A year
        month (int): A month
        day (int): A day
        number_of_week (int, optional): A number of weeks to generate. Defaults to 52.

    Returns:
        list: A list of weekly datetimes.
    """
    if not isinstance(year, int) or not isinstance(month, int) or not isinstance(day, int):
        raise ValueError(f"year {year}, month {month}, and day {day} must all be integers.")
    first_date = datetime(year, month, day)
    week_list = [first_date+i*timedelta(weeks=1) for i in range(number_of_week)]
    return week_list

def monthly_date_list(year, month, day, number_of_month=12):
    """Generate a list of monthly dates.

    Args:
        year (int): A initial year
        month (month): A start month.
        day (int): A start day.
        number_of_month (int, optional): A number of months to generate. Defaults to 12.

    Returns:
        lsit: A list of monthly dates.
    """
    first_date = datetime(year, month, day)
    month_list = [first_date+relativedelta(month=i) for i in range(1, number_of_month+1)]
    return month_list

def yearly_date_list(year, month, day, number_of_year=10):
    """Generate a list of years.

    Args:
        year (int): A start year.
        month (int): A start month.
        day (int): A start day.
        number_of_year (int, optional): A number of years to generate. Defaults to 10.

    Returns:
        list: A list of generated years.
    """
    start_date = datetime(year, month, day)
    year_list = [start_date.replace(year=start_date.year + i) for i in range(number_of_year)]
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
    if len(col)==len(date_list):
        col["time"] = date_list
    return col 

