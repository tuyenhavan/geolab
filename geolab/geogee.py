import ee

from geolab import commons


def index_join(collection_a, collection_b, property_name):
    """
    Joins two image collections on their 'system:index' property.

    Args:
    - collection_a: ee.ImageCollection, the primary collection.
    - collection_b: ee.ImageCollection, the secondary collection.
    - property_name: str, the name of the property that references the joined image.

    Returns:
    - ee.ImageCollection: The joined and merged image collection.
    """
    # Perform the join on 'system:index'
    joined = ee.ImageCollection(
        ee.Join.saveFirst(property_name).apply(
            {
                "primary": collection_a,
                "secondary": collection_b,
                "condition": ee.Filter.equals(
                    leftField="system:index", rightField="system:index"
                ),
            }
        )
    )

    # Merge the bands of the joined image
    def merge_bands(image):
        joined_image = ee.Image(image.get(property_name))
        return image.addBands(joined_image)

    return joined.map(merge_bands)


def modis_cloud_mask(col, from_bit, to_bit, qa_band="DetailedQA", threshold=1):
    """Return a collection of MODIS cloud-free images

    Args:
        col (ee.ImageCollection): The input image collection.
        from_bit (int): The start bit to extract.
        to_bit (int): The last bit to extract.
        QA_band (str|optional): The quality band which contains cloud-related infor. Default to DetailedQA.
        threshold (int|optional): The threshold value to mask cloud-related pixels. Default to 1.

    Returns:
        ee.ImageCollection: The output collection with cloud-free pixels.
    """
    if not isinstance(col, ee.ImageCollection):
        raise TypeError("Unsupported data type. It only supports ee.ImageCollection")
    out_col = commons.cloud_mask(col, from_bit, to_bit, qa_band, threshold)
    return out_col


def landsat_cloud_mask(col, from_bit, to_bit, qa_band="QA_PIXEL ", threshold=1):
    """Return a collection of Landsat cloud-free images

    Args:
        col (ee.ImageCollection): The input image collection.
        from_bit (int): The start bit to extract.
        to_bit (int): The last bit to extract.
        QA_band (str|optional): The quality band which contains cloud-related infor. Default to QA_PIXEL.
        threshold (int|optional): The threshold value to mask cloud-related pixels. Default to 1.

    Returns:
        ee.ImageCollection: The output collection with cloud-free pixels.
    """
    if not isinstance(col, ee.ImageCollection):
        raise TypeError("Unsupported data type. It only supports ee.ImageCollection")
    out_col = commons.cloud_mask(col, from_bit, to_bit, qa_band, threshold)
    return out_col


def sentinel2_cloud_mask(col, from_bit, to_bit, qa_band="QA60", threshold=0):
    """Return a collection of Sentinel-2 cloud-free images

    Args:
        col (ee.ImageCollection): The input image collection.
        from_bit (int): The start bit to extract (e.g., 10)
        to_bit (int): The last bit to extract (e.g., 11)
        QA_band (str|optional): The quality band which contains cloud-related infor. Default to DetailedQA.
        threshold (int|optional): The threshold value to mask cloud-related pixels. Default to 1.

    Returns:
        ee.ImageCollection: The output collection with cloud-free pixels.
    """
    if not isinstance(col, ee.ImageCollection):
        raise TypeError("Unsupported data type. It only supports ee.ImageCollection")
    out_col = commons.cloud_mask(col, from_bit, to_bit, qa_band, threshold)
    return out_col


def annual_composite(
    ds, month_list=None, aoi=None, start_year=2018, end_year=2022, method="mean"
):
    """Aggregating annual data with selected months of a year from ImageCollection.


    Args:
        ds (ee.ImageCollection): An ImageCollection used to aggregate annual values.
        month_list (list): A list of selected months for annual aggregation.
        aoi (ee.FeatureCollection|ee.Geometry|ee.Feature, optional): A area of interest to clip. Defaults to None.
        start_year (int, optional): A starting year. Defaults to 2018.
        end_year (int, optional): An ending year. Defaults to 2022.
        cal_method (str, optional): A annual aggregation method. Defaults to "mean".

    Raises:
        TypeError: Raise errors if not correct data types provided.
        TypeError: Raise errors if not correct data types provided.

    Returns:
        ee.ImageCollection: An annual aggregated ImageCollection (e.g., mean for 2018, 2019, 2020, etc)
    """

    if isinstance(month_list, type(None)):
        month_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    if not isinstance(ds, ee.ImageCollection):
        raise TypeError("Please provide ee.ImageCollection.")
    if isinstance(month_list, list):
        month_cols = ee.List(month_list).map(
            lambda m: ds.filter(ee.Filter.calendarRange(m, m, "month"))
        )
    else:
        raise ValueError("Please provide month list.")
    if isinstance(method, str):
        if method.lower() in ["total", "sum"]:
            cal_dict = commons.CollectionReducer().sum
        elif method.lower() in ["max", "maximum"]:
            cal_dict = commons.CollectionReducer().max
        elif method.lower() in ["min", "minimum"]:
            cal_dict = commons.CollectionReducer().min
        elif method.lower() in ["median"]:
            cal_dict = commons.CollectionReducer().median
        elif method.lower() in ["std"]:
            cal_dict = commons.CollectionReducer().std
        else:
            cal_dict = commons.CollectionReducer().mean
    else:
        raise ValueError("Unsupported method.")
    # Convert list of ImageCollection into one single ImageCollection
    tem_cols = ee.ImageCollection(ee.FeatureCollection(month_cols).flatten()).sort(
        "system:time_start"
    )
    # Get the list of years
    year_list = ee.List.sequence(start_year, end_year)

    annual_col = ee.ImageCollection.fromImages(
        year_list.map(
            lambda y: tem_cols.filter(ee.Filter.calendarRange(y, y, "year"))
            .reduce(**cal_dict)
            .set({"system:time_start": ee.Date.fromYMD(y, 1, 1)})
        )
    )
    if aoi is not None and isinstance(
        aoi, (ee.FeatureCollection, ee.Geometry, ee.Feature)
    ):
        annual_col = annual_col.map(lambda img: img.clip(aoi))
    return annual_col


def monthly_composite(ds, method="mean"):
    """Create a monthly composite from ImageCollection using mean or max or median and min aggregation methods.

    Args:
        ds (ee.ImageCollection): An ImageCollection contains daily/multiple days time series
        method (str, optional): A method for aggregating data. Defaults to "mean".

    Raises:
        TypeError: Raise an error if ds is not an instance of ee.ImageCollection

    Returns:
        ee.ImageCollection: An monthly composite collection.
    """
    if not isinstance(ds, ee.ImageCollection):
        raise TypeError("Please provide ee.ImageCollection.")

    start_date = ee.Date(ds.first().get("system:time_start"))
    last_date = ee.Date(
        ds.sort("system:time_start", False).first().get("system:time_start")
    )

    month_totals = last_date.difference(start_date, "month").round()
    month_list = ee.List.sequence(0, month_totals)

    def monthly_value(m):
        current_date = start_date.advance(m, "month")
        m = current_date.get("month")
        y = current_date.get("year")
        subcol = ds.filter(ee.Filter.calendarRange(m, m, "month")).filter(
            ee.Filter.calendarRange(y, y, "year")
        )
        subsize = subcol.size()

        if method.lower().strip() in ["max", "total"]:
            img = subcol.sum()
        elif method.lower().strip() in ["min"]:
            img = subcol.min()
        elif method.lower().strip() in ["median"]:
            img = subcol.median()
        else:
            img = subcol.mean()
        img = img.set({"system:time_start": ee.Date.fromYMD(y, m, 1).millis()})
        return ee.Algorithms.If(subsize.gt(0), img)

    fcols = ee.ImageCollection.fromImages(month_list.map(monthly_value))
    return fcols


def weekly_composite(ds, method="mean"):
    """Generate weekly image using the provided mode"""
    assert isinstance(ds, ee.ImageCollection), "Please provide ee.ImageCollection"
    start_date = ee.Date(ds.first().get("system:time_start"))
    last_date = ee.Date(
        ds.sort("system:time_start", False).first().get("system:time_start")
    )
    week_difference = (
        start_date.advance(1, "week").millis().subtract(start_date.millis())
    )
    weeklist = ee.List.sequence(
        start_date.millis(), last_date.millis(), week_difference
    )

    def compute(w):
        first = ee.Date(w)
        last = first.advance(1, "week")
        subcol = ds.filterDate(first, last)
        if method.lower().strip() in ["median"]:
            mode_task = commons.CollectionReducer().median
        elif method.lower().strip() in ["max", "maximum"]:
            mode_task = commons.CollectionReducer().max
        elif method.lower().strip() in ["min", "minimum"]:
            mode_task = commons.CollectionReducer().min
        else:
            mode_task = commons.CollectionReducer().mean
        img = subcol.reduce(**mode_task).set({"system:time_start": first.millis()})
        return ee.Algorithms.If(subcol.size().gt(2), img)

    fcol = ee.ImageCollection.fromImages(weeklist.map(compute))
    return fcol


def daily_composite(ds, method="max"):
    """Aggregate data from hourly to daily composites

    Args:
        ds (ImageCollection): The input image collection.
        mode (str|optional): Aggregated modes [max, min, mean, median, sum]. Default to max.

    Return:
        ImageCollection: The daily composite
    """
    if isinstance(method, str):
        method = method.lower().strip()

    # Get the starting and ending dates of the collection
    start_date = ee.Date(
        ee.Date(ds.first().get("system:time_start")).format("YYYY-MM-dd")
    )
    end_date = ee.Date(
        ee.Date(
            ds.sort("system:time_start", False).first().get("system:time_start")
        ).format("YYYY-MM-dd")
    )

    # Get the number of days
    daynum = end_date.difference(start_date, "day")
    slist = ee.List.sequence(0, daynum)
    date_list = slist.map(lambda i: start_date.advance(i, "day"))

    def sub_col(date_input):
        first_date = ee.Date(date_input)
        last_date = first_date.advance(1, "day")
        subcol = ds.filterDate(first_date, last_date)
        size = subcol.size()

        if method in ["max", "maximum"]:
            mode_task = commons.CollectionReducer().max
        elif method in ["mean", "average"]:
            mode_task = commons.CollectionReducer().mean
        elif method in ["min", "minimum"]:
            mode_task = commons.CollectionReducer().min
        elif method in ["median"]:
            mode_task = commons.CollectionReducer().median
        elif method in ["sum", "total"]:
            mode_task = commons.CollectionReducer().sum
        else:
            raise ValueError("Unsupported method.")

        img = subcol.reduce(**mode_task).set({"system:time_start": first_date.millis()})

        return ee.Algorithms.If(size.gt(0), img)

    new_col = ee.ImageCollection.fromImages(date_list.map(sub_col))
    return new_col


def calculate_vci(ds):
    """Calculating monthly VCI based from monthly NDVI time series.

    Args:
        ds (ee.ImageCollection): An ImageCollection of monthly NDVI

    Returns:
        ee.ImageCollection: An ImageCollection of monthly VCI.
    """

    month_list = ee.List.sequence(1, 12)

    def vci_subcol(m):
        ndvi = ds.filter(ee.Filter.calendarRange(m, m, "month"))
        min_ndvi = ndvi.min()
        max_ndvi = ndvi.max()
        ket = ndvi.map(
            lambda img: img.subtract(min_ndvi)
            .divide(max_ndvi.subtract(min_ndvi))
            .rename("VCI")
            .copyProperties(img, ["system:time_start"])
        )
        return ket

    fcols = ee.ImageCollection(
        ee.FeatureCollection(month_list.map(vci_subcol)).flatten()
    ).sort("system:time_start")
    return fcols


def chunk_maker(feature_col, ncols, nrows):
    """Split the study area into different chunks to facilitate the computation.

    Args:
        feature_col (ee.FeatureCollection): A region of interest
        ncols (n): The number of columns
        nrows (n): The number of rows

    Return:
        FeatureCollection: The number of chunks covers the study area.
    """
    if isinstance(feature_col, ee.FeatureCollection):
        data = feature_col
    else:
        raise TypeError("Data must be ee.FeatureCollection!")

    def get_bound(feature_col):
        """Get the min, max of the longitude and latitude of the study area

        Return:
            list: [(min_long, max_long), (min_lat, max_lat)]

        """
        bbox = feature_col.geometry().bounds().coordinates().getInfo()[0]
        min_long = min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
        max_long = max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])

        min_lat = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
        max_lat = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])

        return [
            (round(min_long), round(max_long + 1)),
            (round(min_lat - 1), round(max_lat + 1)),
        ]

    bbox = get_bound(data)
    min_lon, max_lon = bbox[0]
    min_lat, max_lat = bbox[1]
    # Space or distance of chunks
    lon_dist = (max_lon - min_lon) / ncols
    lat_dist = (max_lat - min_lat) / nrows

    polys = []
    cell = 0
    for lon in commons.arange(min_lon, max_lon, lon_dist):
        x1 = lon
        x2 = lon + lon_dist
        for lat in commons.arange(min_lat, max_lat, lat_dist):
            cell += 1
            y1 = lat
            y2 = lat + lat_dist
            polys.append(
                ee.Feature(ee.Geometry.Rectangle(x1, y1, x2, y2), {"label": cell})
            )
    feat_polys = ee.FeatureCollection(polys)
    grid = feat_polys.filterBounds(feature_col)
    index_list = ee.List.sequence(0, grid.size().subtract(1))
    flist = grid.toList(grid.size())
    final_col = index_list.map(
        lambda i: ee.Feature(flist.get(i)).set("system:index", ee.Number(i).format())
    )
    final_col = ee.FeatureCollection(final_col)

    return final_col


def export_to_googledrive(
    ds, aoi, folder_name="GEE_Data", file_name="NDVI_data", res=1000
):
    """Export an image from GEE with a given scale and area of interest
    to the Google Drive. If input data is an ImageCollection, it will convert it
    into an image and then export. The collection should contains only single data,
    for example NDVI bands or precipitation bands or LST bands.

        Args:
            ds (ee.Image|ee.ImageCollection): The input ee.Image or ee.ImageCollection
            aoi (FeatureCollection): The area of interest to clip the images.
            folder_name (str): An output file name. Default is GEE_Data
            res (int): A spatial resolution in meters. Default is 1km.

        Returns:
            ee.Image: the clipped image with crs: 4326
    """
    if isinstance(aoi, (ee.geometry.Geometry, list)):
        aoi = ee.Geometry.Polygon(aoi)
    if isinstance(ds, ee.ImageCollection):
        # Convert it to an image
        img = ds.toBands()
        # get bands and rename
        oldband = img.bandNames().getInfo()
        newband = ["_".join(i.split("_")[::-1]) for i in oldband]
        # Rename it
        new_img = img.select(oldband, newband).clip(aoi)
    elif isinstance(ds, ee.Image):
        new_img = ds.clip(aoi)
    else:
        raise TypeError("Unsupported data type!")
    if isinstance(aoi, ee.featurecollection.FeatureCollection):
        aoi = aoi.geometry().bounds().getInfo()["coordinates"]
    # Initialize the task of downloading an image
    task = ee.batch.Export.image.toDrive(
        image=new_img,  # an ee.Image object.
        # an ee.Geometry object.
        region=aoi,
        description=folder_name,
        folder=folder_name,
        fileNamePrefix=file_name,
        crs="EPSG:4326",
        scale=res,
        maxPixels=1e13,
    )
    task.start()


def export_to_asset(
    ds, aoi, assetId, description="Exported_Data_To_Asset", res=1000, crs=None
):
    """Export an image from GEE with a given scale and area of interest
    to the Google Drive. If input data is an ImageCollection, it will convert it
    into an image and then export. The collection should contains only single data,
    for example NDVI bands or precipitation bands or LST bands.

        Args:
            ds (ee.Image|ee.ImageCollection): The input ee.Image or ee.ImageCollection
            aoi (FeatureCollection): The area of interest to clip the images.
            folder_name (str): An output file name. Default is GEE_Data
            res (int): A spatial resolution in meters. Default is 1km.
            crs (str|optional): The output crs. Default to EPSG:4326

        Returns:
            ee.Image: the clipped image with crs: 4326
    """
    if crs is None:
        crs = "EPSG:4326"
    if isinstance(ds, ee.ImageCollection):
        # Convert it to an image
        img = ds.toBands()
        # get bands and rename
        oldband = img.bandNames().getInfo()
        newband = ["_".join(i.split("_")[::-1]) for i in oldband]
        # Rename it
        new_img = img.select(oldband, newband).clip(aoi)
    elif isinstance(ds, ee.Image):
        new_img = ds.clip(aoi)
    else:
        raise TypeError("Unsupported data type!")

    # Initialize the task of downloading an image
    task = ee.batch.Export.image.toAsset(
        image=new_img,  # an ee.Image object.
        # an ee.Geometry object.
        region=aoi.geometry().bounds().getInfo()["coordinates"],
        description=description,
        assetId=assetId,
        crs=crs,
        scale=res,
    )
    task.start()

def mask_landsat_clouds(collection):
    """
    Applies a cloud and cloud shadow mask to a Landsat ImageCollection using the QA_PIXEL band.
    
    The function removes pixels flagged as clouds or cloud shadows based on the QA_PIXEL band
    in Landsat imagery. This function mainly works with Landsat 8 and 9 level 2 collection 2 tier 1 data.

    Parameters:
        collection (ee.ImageCollection): The input Landsat ImageCollection.

    Returns:
        ee.ImageCollection: The masked ImageCollection with clouds and cloud shadows removed.

    Example:
        # Load a Landsat 8 ImageCollection
        landsat_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
                                .filterBounds(ee.Geometry.Point([106.85, 10.76])) \
                                .filterDate("2023-01-01", "2023-12-31")

        # Apply the cloud masking function
        masked_collection = mask_landsat_clouds(landsat_collection)
    """
    def mask_clouds(image):
        """Masks clouds and cloud shadows in a Landsat image using the QA_PIXEL band."""
        cloud_shadow_bit = 1 << 3  # Bit 3: Cloud shadow
        cloud_bit = 1 << 5         # Bit 5: Cloud

        qa = image.select("QA_PIXEL")
        mask = qa.bitwiseAnd(cloud_shadow_bit).eq(0).And(
            qa.bitwiseAnd(cloud_bit).eq(0)
        )
        return image.updateMask(mask)
    return collection.map(mask_clouds)

def convert_landsat_lst_to_celsius(collection, roi=None, band="ST_10"):
    """
    Converts Landsat surface temperature (LST) from Kelvin to Celsius in an ImageCollection.

    The function applies a cloud mask using `mask_landsat_clouds`, optionally filters by 
    a region of interest (ROI), and converts the specified thermal band from Kelvin to Celsius.

    Parameters:
        collection (ee.ImageCollection): The input Landsat ImageCollection containing LST data.
        roi (ee.Geometry, optional): A region of interest to filter the collection. Defaults to None.
        band (str, optional): The thermal band name to process. Defaults to "ST_10" (Landsat 8/9).

    Returns:
        ee.ImageCollection: The processed ImageCollection with the LST band converted to Celsius.

    Example:
        # Define a region of interest (ROI)
        roi = ee.Geometry.Point([106.85, 10.76])

        # Load a Landsat 8 Collection
        landsat_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
                                .filterDate("2023-01-01", "2023-12-31")

        # Convert LST to Celsius
        lst_celsius_collection = convert_landsat_lst_to_celsius(landsat_collection, roi)

        # Display the first image
        Map.addLayer(lst_celsius_collection.first(), {"min": 20, "max": 50, "palette": ["blue", "green", "red"]}, "LST in Celsius")
    """
    if roi:
        collection = collection.filterBounds(roi)

    # Apply cloud masking
    collection = mask_landsat_clouds(collection)

    def to_celsius(image):
        """Converts the specified thermal band from Kelvin to Celsius."""
        lst_celsius = image.select(band).multiply(0.00341802).subtract(273.15) \
                           .rename("LST_Celsius")  # Rename for clarity
        return image.addBands(lst_celsius).copyProperties(image, ["system:time_start"])

    return collection.map(to_celsius)