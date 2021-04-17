from os import listdir
from os.path import isfile, join
from polygon_matrix_transformer import PolygonMatrixTransformer
from geojson_reader import GeoJsonReader
import pandas as pd
from shapely.geometry import Polygon
import numpy as np
from PIL import Image, ImageDraw
from geojson_to_pixel_transformer import GeoJsonToPixelTransformer
import argparse



def get_image_metadata_dataframe(path: str):
    return pd.read_csv(path, sep="\t")


def get_image_corners(img_metadata):
    to_modify = 0.005
    # top left corner first, then clockwise
    return Polygon([(img_metadata.longitude - to_modify, img_metadata.latitude + to_modify),
            (img_metadata.longitude + to_modify, img_metadata.latitude + to_modify),
            (img_metadata.longitude + to_modify, img_metadata.latitude - to_modify),
            (img_metadata.longitude - to_modify, img_metadata.latitude - to_modify),
            (img_metadata.longitude - to_modify, img_metadata.latitude + to_modify)])


def get_intersections_with_polygons(polygon_for_intersection: Polygon, polygons_array: "list[Polygon]"):
    intersections = []
    for polygon in polygons_array:
        intersection = polygon_for_intersection.intersection(polygon)
        if not intersection.is_empty:
            intersections.append(intersection)
    return intersections


def get_class_color(index: int, transparent=False):
    transparency = 125 if transparent else 255
    colors_dict = {
        0: (255, 0, 0, transparency),
        1: (0, 0, 255, transparency),
        2: (0, 255, 0, transparency)
    }
    return colors_dict[index]


if __name__ == "__main__":
    # python main.py --path='F:\Download\WD_dane' --result='F:\Download\WD_dane\Out_Results' --debug=True
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to all data', required=True)
    parser.add_argument('--result', help='Path where results should be produced', required=True)
    parser.add_argument('--debug', help='Enable image production for debugging purposes')

    args = args=parser.parse_args()

    debug_mode = args.debug != None and args.debug != False

    data_path = args.path
    results_path = args.result

    images_folder_path = f'{data_path}/Photos/200MSDCF'
    metadata_file_path = f'{data_path}/Photos/EOZ_lot1_WL_RPY_Hgeoid.txt'

    geo_json_files_folder_path = f'{data_path}/Classes'
    geo_json_files = [f for f in listdir(geo_json_files_folder_path) if isfile(join(geo_json_files_folder_path, f)) and f.endswith(".geojson")] 

    image_metadata = get_image_metadata_dataframe(metadata_file_path)
    data_classes_as_polygons = [GeoJsonReader().load(join(geo_json_files_folder_path, path)) for path in geo_json_files]

    for img_filename in listdir(images_folder_path): 
        img_full_path = join(images_folder_path, img_filename)
        if not isfile(img_full_path) or not img_filename.endswith(".JPG"):
            continue

        img = Image.open(img_full_path)
        if debug_mode:
            img = img.convert("RGBA")

        metadata_of_img_to_process = image_metadata[image_metadata.Filename == img_filename].iloc[0]
        img_polygon = get_image_corners(metadata_of_img_to_process)

        intersections_with_data_classes = [get_intersections_with_polygons(img_polygon, data_class_polygon) for data_class_polygon in data_classes_as_polygons]

        object_transformer = PolygonMatrixTransformer()
        
        geojson_points = object_transformer.transform_to_matrix(img_polygon)
        img_points = np.array([
            [0, img.size[0] - 1, img.size[0] - 1, 0],
            [0, 0, img.size[1] - 1, img.size[1] - 1],
            [1, 1, 1, 1]
        ])

        coords_transformer = GeoJsonToPixelTransformer(geojson_points, img_points)

        train_image = Image.new('RGBA', (img.size[0], img.size[1]))
        brush = ImageDraw.Draw(train_image)

        is_any_intersection = False
        for index, intersections_with_data_class  in enumerate(intersections_with_data_classes):
            color = get_class_color(index, debug_mode)

            for intersection in intersections_with_data_class:
                is_any_intersection = True
                intersection_matrix_pixels = coords_transformer.transform(object_transformer.transform_to_matrix(intersection))
                intersection_pixels = zip(intersection_matrix_pixels[0], intersection_matrix_pixels[1])
                
                brush.polygon(list(intersection_pixels), fill=color)

        if is_any_intersection:
            if debug_mode:
                train_image = Image.alpha_composite(img, train_image)
            train_image.save(f"{results_path}/out_{img_filename[:-4]}.png")
