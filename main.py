from os import listdir
from os.path import isfile, join
from p_matrix_parser import PmatrixParser
from polygon_matrix_transformer import PolygonMatrixTransformer
from geojson_reader import GeoJsonReader
import pandas as pd
from shapely.geometry import Polygon
import numpy as np
from PIL import Image, ImageDraw
import argparse
import datetime


def print_message_with_time(message: str):
    print(f'[{datetime.datetime.now().strftime("%H:%M:%S")}] {message}')


def get_image_metadata_dataframe(path: str):
    return pd.read_csv(path, sep="\t")


def get_intersections_with_polygons(polygon_for_intersection: Polygon, polygons_array: "list[Polygon]"):
    intersections = []

    count = 0
    for polygon in polygons_array:
        try:
            intersection = polygon_for_intersection.intersection(polygon)
            if not intersection.is_empty:
                intersections.append(intersection)
        except Exception as e:
            count += 1
            print(f'Exception: {e}')
    print(f"Number of exceptions during calculating intersections {count}")
    return intersections


def get_class_color(index: int, transparent=False):
    transparency = 125 if transparent else 255
    colors_dict = {
        0: (255, 0, 0, transparency),
        1: (0, 0, 255, transparency),
        2: (0, 255, 0, transparency)
    }
    return colors_dict[index]


def transform_to_pixel_coordinates(p_matrix, x: float, y: float, z: float):
    res = np.dot(p_matrix, np.array([x, y, z, 1]).T)
    return (res[0] / res[2], res[1] / res[2])


def load_offset(path: str):
    with open(path, 'r') as file:
        return np.array([float(x) for x in file.readline().split(' ')])


def get_polygon_in_pixel_coordinates(polygon: Polygon, p_matrix, height: float):
    polygon_coordinates = []
    for x, y in zip(polygon.exterior.coords.xy[0], polygon.exterior.coords.xy[1]):
        new_coordinates = transform_to_pixel_coordinates(p_matrix, x, y, height)
        polygon_coordinates.append(new_coordinates)
    
    return Polygon(polygon_coordinates)


def transform_polygons_for_classes(classes_as_polygons: "list[list[Polygon]]", p_matrix, offset):
    sea_level = 103
    height = sea_level - offset[2]

    new_polygons_classes = []
    for class_as_polygons in classes_as_polygons:
        new_polygon_class = []
        for polygon in class_as_polygons:
            new_polygon = get_polygon_in_pixel_coordinates(polygon, p_matrix, height)
            new_polygon_class.append(new_polygon)
        new_polygons_classes.append(new_polygon_class)
    
    return new_polygons_classes


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
    pmatrix_path = f'{data_path}/Processing/07_05_2020_WL/1_initial/params/07_05_2020_WL_pmatrix.txt'
    offset_path = f'{data_path}/Processing/07_05_2020_WL/1_initial/params/07_05_2020_WL_offset.xyz'


    print_message_with_time('------ Starting program ------')

    geo_json_files_folder_path = f'{data_path}/Classes'
    geo_json_files = [f for f in listdir(geo_json_files_folder_path) if isfile(join(geo_json_files_folder_path, f)) and f.endswith(".geojson")] 

    print_message_with_time('------ Starting metadata parsing ------')

    offset = load_offset(offset_path)
    images_metadata = get_image_metadata_dataframe(metadata_file_path)
    data_classes_as_polygons = [GeoJsonReader().load(join(geo_json_files_folder_path, path), offset) for path in geo_json_files]
    dictionary_with_p_matrices = PmatrixParser().parse(pmatrix_path)

    print_message_with_time('------ Finished metadata parsing ------')
    print_message_with_time('------ Starting image processing ------')

    exceptions_count = 0
    image_index = 0
    images_names = listdir(images_folder_path)
    for img_filename in images_names:
        image_index += 1

        try:
            print_message_with_time(f'\n--- Processing image {img_filename} [{image_index}/{len(images_names)}] ---')

            img_full_path = join(images_folder_path, img_filename)
            if not isfile(img_full_path) or not img_filename.endswith(".JPG"):
                print('Bad image file')
                continue

            metadata_of_img_to_process = images_metadata[images_metadata.Filename == img_filename].iloc[0]
            roll = abs(metadata_of_img_to_process['roll[deg]'])
            pitch = abs(metadata_of_img_to_process['pitch[deg]'])
            
            if roll > 10 or pitch > 10:
                print('Roll > 10 or pitch > 10')
                continue

            img = Image.open(img_full_path)
            if debug_mode:
                img = img.convert("RGBA")

            p_matrix = dictionary_with_p_matrices.get(img_filename)
            if p_matrix is None:
                print('Pmatrix is None')
                continue

            img_polygon = Polygon([
                [0, 0],
                [img.size[0], 0],
                [img.size[0], img.size[1]],
                [0, img.size[1]],
                [0, 0]
            ])

            data_classes_as_polygons_copy = transform_polygons_for_classes(data_classes_as_polygons, p_matrix, offset)
            intersections_with_data_classes = [get_intersections_with_polygons(img_polygon, data_class_polygon) for data_class_polygon in data_classes_as_polygons_copy]

            train_image = Image.new('RGBA', (img.size[0], img.size[1]))
            brush = ImageDraw.Draw(train_image)

            is_any_intersection = False
            for index, intersections_with_data_class  in enumerate(intersections_with_data_classes):
                color = get_class_color(index, debug_mode)

                for intersection in intersections_with_data_class:
                    is_any_intersection = True
                    intersection_matrix_pixels = PolygonMatrixTransformer().transform_to_matrix(intersection)
                    intersection_pixels = list(zip(intersection_matrix_pixels[0], intersection_matrix_pixels[1]))

                    brush.polygon(intersection_pixels, fill=color)

            if is_any_intersection:
                if debug_mode:
                    train_image = Image.alpha_composite(img, train_image)
                path = f"{results_path}/out{'_debug' if debug_mode else ''}_{img_filename[:-4]}.png"
                train_image.save(path)
            else:
                print('No intersections in current image')
        except Exception as e:
            exceptions_count += 1
            print(f'!!!!!! Processing image {img_filename} FAILED: {e}')

    print(f'\n\nFinal exceptions count: {exceptions_count}')
    print_message_with_time('------ Finished image processing ------')
