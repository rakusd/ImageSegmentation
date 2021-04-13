from geojson_reader import GeoJsonReader
import pandas as pd
from shapely.geometry import Polygon
import numpy as np
from PIL import Image


def get_image_metadata_dataframe(path: str):
    return pd.read_csv(path, sep=" ")


def get_image_corners(img_metadata):
    to_modify = 0.005
    # top left corner first, then clockwise
    return Polygon([(img_metadata.longitude - to_modify, img_metadata.latitude + to_modify),
            (img_metadata.longitude + to_modify, img_metadata.latitude + to_modify),
            (img_metadata.longitude + to_modify, img_metadata.latitude - to_modify),
            (img_metadata.longitude - to_modify, img_metadata.latitude - to_modify),
            (img_metadata.longitude - to_modify, img_metadata.latitude + to_modify)])


if __name__ == "__main__":
    geo_json_file_path = "data/geojsons/klasa_2.geojson"
    metadata_file_path = "data/coordinates/EOZ_lot1_WL_RPY_Helips.txt"
    img_name = "DSC01170.JPG"
    img_path = f'data/img/{img_name}'

    img = Image.open(img_path)

    image_metadata = get_image_metadata_dataframe(metadata_file_path)
    polygons = GeoJsonReader().load(geo_json_file_path)

    metadata_of_img_to_process = image_metadata[image_metadata.Filename == img_name].iloc[0]
    img_polygon = get_image_corners(metadata_of_img_to_process)

    intersections = []
    for polygon in polygons:
        intersection = img_polygon.intersection(polygon)
        if not intersection.is_empty:
            intersections.append(intersection)

    A = np.array([[point[0], point[1]] for point in zip(img_polygon.exterior.coords.xy[0], img_polygon.exterior.coords.xy[1])][:-1])
    B = np.array([
        [0, 0],
        [img.size[0] - 1, 0],
        [img.size[0] - 1, img.size[1] - 1],
        [0, img.size[1] - 1]
    ])

    result = np.linalg.inv(A).dot(B)
    print(result)
    #print(img_polygon)
    