import json
from shapely.geometry import Polygon
from pyproj import Transformer


class GeoJsonReader:
    def load(self, path: str, offset) -> "list[Polygon]":
        with open(path) as json_file:
            geo_json = json.load(json_file)

        polygons = []
        for feature in geo_json["features"]:
            coordinates = feature["geometry"]["coordinates"]
            if (len(coordinates) == 0 or len(coordinates[0]) == 0):
                continue
            polygon_array = coordinates[0][0]
            transformed_polygon_array = []
            for polygon_point in polygon_array:
                transformed_coordinates = self.transform_wgs_to_epsg_2178(polygon_point[0], polygon_point[1])
                transformed_coordinates -= offset[:-1]
                transformed_polygon_array.append(transformed_coordinates)

            polygon = Polygon(transformed_polygon_array)
            polygons.append(polygon)

        return polygons
    
    def transform_wgs_to_epsg_2178(self, longitude: float, latitude: float):
        transformer = Transformer.from_crs(4326, 2178)
        result = transformer.transform(latitude, longitude)
        return (result[1], result[0])