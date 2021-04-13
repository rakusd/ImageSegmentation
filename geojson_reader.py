import json
from shapely.geometry import Polygon

class GeoJsonReader:
    def load(self, path: str) -> "list[Polygon]":
        with open(path) as json_file:
            geo_json = json.load(json_file)

        polygons = []
        for feature in geo_json["features"]:
            polygon_array = feature["geometry"]["coordinates"][0][0]
            polygon = Polygon(polygon_array)
            polygons.append(polygon)

        return polygons
