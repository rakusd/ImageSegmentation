from shapely.geometry import Polygon
import numpy as np

class PolygonMatrixTransformer:
    def transform_to_matrix(self, polygon: Polygon):
        x_coords = polygon.exterior.coords.xy[0][:-1]
        y_coords = polygon.exterior.coords.xy[1][:-1]
        
        coords_num = len(x_coords)
        z_coords = np.ones(coords_num)
        
        matrix = np.array([x_coords, y_coords, z_coords])
        
        return matrix

    def transform_to_polygon(self, matrix: np.ndarray):
        polygon_points = matrix[0:2].transpose()
        polygon_points = np.concatenate((polygon_points, [polygon_points[0]]))
        
        polygon = Polygon(polygon_points)

        return polygon