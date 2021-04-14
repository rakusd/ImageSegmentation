from shapely.geometry import Polygon
import numpy as np

class GeoJsonToPixelTransformer:
    def __init__(self, geojson_matrix: np.ndarray, pixel_matrix: np.ndarray):
        # numpy operates on square matrices
        pixel_matrix = pixel_matrix[:, :3]
        geojson_matrix = geojson_matrix[:, :3]

        self._transformation_matrix = pixel_matrix.dot(np.linalg.inv(geojson_matrix))


    def transform(self, matrix: np.ndarray):
        return self._transformation_matrix.dot(matrix)