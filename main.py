from polygon_matrix_transformer import PolygonMatrixTransformer
from geojson_reader import GeoJsonReader
import pandas as pd
from shapely.geometry import Polygon
import numpy as np
from PIL import Image, ImageDraw
from geojson_to_pixel_transformer import GeoJsonToPixelTransformer


def get_image_metadata_dataframe(path: str):
    return pd.read_csv(path, sep=" ")


def get_image_corners(img_metadata):
    # TODO: Get exact sea level measurement
    sea_level_point = 103
    drone_height_above_sea_level = img_metadata.alt_geo
    drone_height = drone_height_above_sea_level - sea_level_point

    # 0 = pointing north, going clockwise
    yaw = img_metadata['yaw[deg]']

    # 0 = straight, right wing going lower
    roll = img_metadata['roll[deg]']

    # 0 = straight, plane pointing to the sky
    pitch = img_metadata['pitch[deg]']

    to_modify = 0.005
    # top left corner first, then clockwise
    return Polygon([(img_metadata.longitude - to_modify, img_metadata.latitude + to_modify),
            (img_metadata.longitude + to_modify, img_metadata.latitude + to_modify),
            (img_metadata.longitude + to_modify, img_metadata.latitude - to_modify),
            (img_metadata.longitude - to_modify, img_metadata.latitude - to_modify),
            (img_metadata.longitude - to_modify, img_metadata.latitude + to_modify)])


if __name__ == "__main__":
    geo_json_file_path = "data/geojsons/klasa_2.geojson"
    geo_json_file_path_2 = "data/geojsons/klasa_1.geojson"
    metadata_file_path = "data/coordinates/EOZ_lot1_WL_RPY_Helips.txt"
    img_name = "DSC01170.JPG"
    img_path = f'data/img/{img_name}'

    img = Image.open(img_path)

    image_metadata = get_image_metadata_dataframe(metadata_file_path)
    polygons = GeoJsonReader().load(geo_json_file_path)
    polygons_2 = GeoJsonReader().load(geo_json_file_path_2)

    metadata_of_img_to_process = image_metadata[image_metadata.Filename == img_name].iloc[0]
    img_polygon = get_image_corners(metadata_of_img_to_process)

    intersections = []
    for polygon in polygons:
        intersection = img_polygon.intersection(polygon)
        if not intersection.is_empty:
            intersections.append(intersection)

    intersections_2 = []
    for polygon in polygons_2:
        intersection = img_polygon.intersection(polygon)
        if not intersection.is_empty:
            intersections_2.append(intersection)

    object_transformer = PolygonMatrixTransformer()
    
    geojson_points = object_transformer.transform_to_matrix(img_polygon)
    img_points = np.array([
        [0, img.size[0] - 1, img.size[0] - 1, 0],
        [0, 0, img.size[1] - 1, img.size[1] - 1],
        [1, 1, 1, 1]
    ])

    coords_transformer = GeoJsonToPixelTransformer(geojson_points, img_points)

    transformed_points = coords_transformer.transform(geojson_points)

    train_image = Image.new('RGB', (img.size[0], img.size[1]))
    brush = ImageDraw.Draw(train_image)
    for intersection in intersections:
        intersection_matrix_pixels = coords_transformer.transform(object_transformer.transform_to_matrix(intersection))
        intersection_pixels = zip(intersection_matrix_pixels[0], intersection_matrix_pixels[1])
        
        brush.polygon(list(intersection_pixels), fill="red")

    for intersection in intersections_2:
        intersection_matrix_pixels = coords_transformer.transform(object_transformer.transform_to_matrix(intersection))
        intersection_pixels = zip(intersection_matrix_pixels[0], intersection_matrix_pixels[1])
        
        brush.polygon(list(intersection_pixels), fill="blue")

    train_image.save("out.png")


    print("TEST")
    