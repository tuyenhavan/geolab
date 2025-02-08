import ee


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
