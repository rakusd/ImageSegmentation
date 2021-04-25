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
import math

CAMERA_LENS = 35
SENSOR_WIDTH = 35.9
SENSOR_HEIGHT = 24

EARTH_RADIUS = 6_371_000


def get_image_metadata_dataframe(path: str):
    return pd.read_csv(path, sep="\t")

def get_rotation_matrix(yaw: float, pitch: float, roll:float) -> np.ndarray:

    #yaw, roll = roll, yaw # because we use relative to camera angles not plane

    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    roll = math.radians(roll)

    cos_a = math.cos(yaw)
    cos_b = math.cos(pitch)
    cos_c = math.cos(roll)

    sin_a = math.sin(yaw)
    sin_b = math.sin(pitch)
    sin_c = math.sin(roll)

    return np.array([
        [cos_a * cos_b, cos_a * sin_b * sin_c - sin_a * cos_c, cos_a * sin_b * cos_c + sin_a * sin_c, 0],
        [sin_a * cos_b, sin_a * sin_b * sin_c + cos_a * cos_c, sin_a * sin_b * cos_c - cos_a * sin_c, 0],
        [-sin_b, cos_b * sin_c, cos_b * cos_c, 0],
        [0, 0, 0, 1]
    ])


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

    # https://stackoverflow.com/questions/17599845/camera-projection-on-a-plane
    camera_vertical_angle = 2 * math.atan(SENSOR_HEIGHT/(2 * CAMERA_LENS)) # in radians
    camera_horizontal_angle = 2 * math.atan(SENSOR_WIDTH/(2 * CAMERA_LENS)) # in radians

    camera_rotation_matrix = get_rotation_matrix(yaw, pitch, roll)

    # initial_camera_matrix = np.array([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, -drone_height, 1]
    # ])

    # camera facing up
    # initial_camera_matrix = np.array([
    #     [0, -1, 0, 0],
    #     [1, 0, 0, 0],
    #     [0, 0, -1, -drone_height],
    #     [0, 0, 0, 1]
    # ])

    # camera facing front of plane
    initial_camera_matrix = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, -1, 0, -283],
        [0, 0, 0, 0]
    ])
    # camera facing down
    # initial_camera_matrix = np.array([
    #     [0, 1, 0, 0],
    #     [1, 0, 0, 0],
    #     [0, 0, 1, -drone_height],
    #     [0, 0, 0, 1]
    # ])

    camera_rotate_below_drone_matrix = get_rotation_matrix(0, 90, 0)
    
    final_camera_matrix = camera_rotate_below_drone_matrix.dot(camera_rotation_matrix.dot(initial_camera_matrix))

    # TO DO MIGHT BE CHANGED TO READ ROWS INSTEAD OF COLUMNS
    right = final_camera_matrix[:3, 0]
    up = final_camera_matrix[:3, 1] 
    away = final_camera_matrix[:3, 2]
    #right = final_camera_matrix[0, :3]
    #up = final_camera_matrix[1, :3] 
    #away = final_camera_matrix[2, :3]


    right_factor = math.tan(camera_horizontal_angle / 2)
    up_factor = math.tan(camera_vertical_angle / 2)

    top_left_vector = away - right_factor * right + up_factor * up # SE
    top_right_vector = away + right_factor * right + up_factor * up # NE
    bottom_left_vector = away - right_factor * right - up_factor * up # SW
    bottom_right_vector = away + right_factor * right - up_factor * up #NW

    # drone height is negative as z-axis is pointing to the ground
    time_top_left =  -(-drone_height) / top_left_vector[2]
    time_top_right = -(-drone_height) / top_right_vector[2]
    time_bottom_left = -(-drone_height) / bottom_left_vector[2]
    time_bottom_right = -(-drone_height) / bottom_right_vector[2]

    if time_top_left < 0 or time_top_right < 0 or time_bottom_left < 0 or time_bottom_right:
        pass

    top_left_diff = top_left_vector[0:2] * time_top_left
    top_right_diff = top_right_vector[0:2] * time_top_right
    bottom_left_diff = bottom_left_vector[0:2] * time_bottom_left
    bottom_right_diff = bottom_right_vector[0:2] * time_bottom_right

    latitude_factor = 1/one_meter_to_latitude()
    longitude_factor = 1/one_meter_to_longitude(img_metadata.latitude)
    
    # top left corner first, then clockwise
    # x is north in RPY coordinates therefore latitutde is added x coords
    # return Polygon([(img_metadata.longitude + longitude_factor * top_left_diff[1], img_metadata.latitude + latitude_factor * top_left_diff[0]),
    #         (img_metadata.longitude + longitude_factor * top_right_diff[1], img_metadata.latitude + latitude_factor * top_right_diff[0]),
    #         (img_metadata.longitude + longitude_factor * bottom_right_diff[1], img_metadata.latitude + latitude_factor * bottom_right_diff[0]),
    #         (img_metadata.longitude + longitude_factor * bottom_left_diff[1], img_metadata.latitude + latitude_factor * bottom_left_diff[0]),
    #         (img_metadata.longitude + longitude_factor * top_left_diff[1], img_metadata.latitude + latitude_factor * top_left_diff[0])])

    return Polygon([(21.278184192715582, 52.191409373713782),
        (21.27893210196283, 52.193098789071541 ), 
        (21.277100379053511, 52.193405046999402 ), 
        (21.276351223018249, 52.191719635142071 ), 
        (21.278184192715582, 52.191409373713782 )])


def get_polygon_reversed(polygon: Polygon, try_index: int) -> Polygon:
    x_coords = polygon.exterior.coords.xy[0][0:4]
    y_coords = polygon.exterior.xy[1][0:4]

    return Polygon([
        (x_coords[(0+try_index)%4], y_coords[(0+try_index)%4]),
        (x_coords[(1+try_index)%4], y_coords[(1+try_index)%4]),
        (x_coords[(2+try_index)%4], y_coords[(2+try_index)%4]),
        (x_coords[(3+try_index)%4], y_coords[(3+try_index)%4])
        ])

def one_meter_to_latitude() -> float:
    return (math.pi * EARTH_RADIUS) / 180
    #return  360 / (2*math.pi * EARTH_RADIUS)

def one_meter_to_longitude(latitude: float) -> float:
    return (math.pi * EARTH_RADIUS * math.cos(math.radians(latitude))) / 180
    #return 360 / (EARTH_RADIUS * 2 * math.pi * math.cos(math.radians(latitude)))

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
    #geo_json_files_folder_path = '/home/rakusd/Desktop/Uczelnia/WB/Projekt/ImageSegmentation/data/geojsons'

    geo_json_files = [f for f in listdir(geo_json_files_folder_path) if isfile(join(geo_json_files_folder_path, f)) and f.endswith(".geojson")] 

    image_metadata = get_image_metadata_dataframe(metadata_file_path)
    data_classes_as_polygons = [GeoJsonReader().load(join(geo_json_files_folder_path, path)) for path in geo_json_files]

    for img_filename in listdir(images_folder_path): 
        #test
        if img_filename != "DSC01170.JPG" and img_filename != "DSC01169.JPG":
            continue
        #test
        img_full_path = join(images_folder_path, img_filename)
        if not isfile(img_full_path) or not img_filename.endswith(".JPG"):
            continue

        img = Image.open(img_full_path)
        if debug_mode:
            img = img.convert("RGBA")

        metadata_of_img_to_process = image_metadata[image_metadata.Filename == img_filename].iloc[0]
        img_polygon = get_image_corners(metadata_of_img_to_process)

        copy_img_polygon = img_polygon
        for try_index in range(1):
            img_polygon = get_polygon_reversed(copy_img_polygon, try_index)
            intersections_with_data_classes = [get_intersections_with_polygons(img_polygon, data_class_polygon) for data_class_polygon in data_classes_as_polygons]

            object_transformer = PolygonMatrixTransformer()
        
            geojson_points = object_transformer.transform_to_matrix(img_polygon)

            # img_points = np.array([ # our format is clockwise
            #     [0, img.size[0] - 1, img.size[0] - 1, 0],
            #     [0, 0, img.size[1] - 1, img.size[1] - 1],
            #     [1, 1, 1, 1]
            # ])

            #if checking with geojson from qgis we need to use anticlockwise
            img_points = np.array([ # geojson uses anticlockwise ordering
                [0, 0, img.size[0] - 1, img.size[0] - 1],
                [0, img.size[1] - 1, img.size[1] - 1, 0],
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
                    intersection_pixels = list(zip(intersection_matrix_pixels[0], intersection_matrix_pixels[1]))

                    brush.polygon(intersection_pixels, fill=color)

            if is_any_intersection:
                if debug_mode:
                    train_image = Image.alpha_composite(img, train_image)
                train_image.save(f"{results_path}/out_{img_filename[:-4]}_{try_index}.png")
                print(f"{results_path}/out_{img_filename[:-4]}_{try_index}.png")

        #test
        if img_filename == "DSC01169.JPG":
            break
        #test
