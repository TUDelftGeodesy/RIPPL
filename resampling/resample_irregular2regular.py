"""
This function does resampling from an irregular grid to a regular grid. However, we assume that the grid itself
is regular in the sense that coordinate values increase in rows and columns.

The algorithm consists of 4 main steps.
1. It selects a rectangular region of the irregular grid that is needed for the interpolation
2. It does a delaunay triangulation for this region.
3. It identifies all the points that are within every triangle from the delaunay triangulation
4. It does an interpolation based on the three corners of these triangles.

If needed a step could be added where the delaunay triangulation is done for different sections first, if we know
that many triangles will be empty. These triangles can than be weeded out first.

"""
import numpy as np
from scipy.spatial import Delaunay

from rippl.resampling.select_input_window import SelectInputWindow
from rippl.orbit_geometry.orbit_coordinates import CoordinateSystem


class Irregular2Regular():

    def __init__(self, input_pixels, input_lines, input_val, coordinates='', s_lin=0, s_pix=0, shape=[1,1], multilooking=[1,1], buf=3):

        if isinstance(coordinates, CoordinateSystem):
            self.s_lin = coordinates.first_line
            self.s_pix = coordinates.first_pixel
            self.shape = coordinates.shape
            self.ml_step = np.array(coordinates.multilook) / np.array(coordinates.oversample)

        self.s_lin = 0
        self.s_pix = 0
        self.shape = shape
        self.ml_step = multilooking
        self.buf = 3

        # We correct for an output grid starting at 0 and steps of 1.
        self.lines = (input_lines - s_lin) / self.ml_step[0]
        self.pixels = (input_pixels - s_pix) / self.ml_step[1]
        
        # Data values
        self.input_values = input_val
        

    def __call__(self, weed=False):

        # Create delaunay triangulation
        delaunay, point_coor = Irregular2Regular.delaunay(self.lines, self.pixels, 0, 0, self.shape, self.buf)

        # Weed out non-used delaunay triangles
        if weed:
            point_coor = Irregular2Regular.weed_triangles(delaunay, point_coor)
            delaunay, point_coor = Irregular2Regular.delaunay(point_coor[:, 0], point_coor[:, 1], 0, 0, self.shape, self.buf)

        # Assign interp points to the triangles
        grid_triangle_id = Irregular2Regular.assign_pixels_to_triangle(delaunay, point_coor, 0, 0, self.shape)
        
        # Apply baricentric interpolation.
        output_values = Irregular2Regular.barycentric_interpolation(delaunay, grid_triangle_id, point_coor,
                                                                    self.lines, self.pixels, self.input_values)

        return output_values

    @staticmethod
    def delaunay(line_coor, pixel_coor, s_lin=0, s_pix=0, shape=[0, 0], buf=3):

        # Start with finding the relevant pixel and line coordinates
        used_pixels = SelectInputWindow.input_irregular_bool_grid(line_coor, pixel_coor, s_lin, s_pix, shape, buf)

        # Create a delaunay triangulation for the obtained grid.
        point_coor = np.concatenate((line_coor[:, None], pixel_coor[:, None]), axis=1)
        delaunay = Delaunay(point_coor)

        return delaunay, point_coor

    @staticmethod
    def weed_triangles(delaunay, point_coor):
        # Remove triangles which are not used in the interpolation. This works only if points are not part of any
        # usable triangle

        triangle_coors = point_coor[delaunay.simplices]

        dem_min_line = np.ceil(np.min(triangle_coors[:, :, 0])).astype(np.int32)
        dem_max_line = np.floor(np.max(triangle_coors[:, :, 0])).astype(np.int32)
        dem_min_pixel = np.ceil(np.min(triangle_coors[:, :, 1])).astype(np.int32)
        dem_max_pixel = np.floor(np.max(triangle_coors[:, :, 1])).astype(np.int32)
        del triangle_coors

        invalid = ~(dem_max_line - dem_min_line == 0) * (dem_max_pixel - dem_min_pixel == 0)
        point_ids = np.unique(point_coor[delaunay.simplices[invalid, :]])
        del dem_min_pixel, dem_max_pixel, dem_min_line, dem_max_line

        # Return the usable points.
        return point_coor[point_ids]

    @staticmethod
    def assign_pixels_to_triangle(delaunay, point_coor, s_lin=0, s_pix=0, shape=[0, 0]):
        # This processing_steps_old links the dem grid boxes to line and pixel coordinates. It returns a sparse matrix with all the
        # pixels and lines inside the grid boxes.

        # For simplicity we assume that there is minimal overlay of the points in the dem grid. This means that
        # triangulation is already known. In fact this is also very likely for low resolution datasets like SRTM,
        # because we barely have areas steeper than the satellite look angle (50-70 degrees), perpendicular to the
        # satellite orbit.

        # For every dem pixel we find the radar pixels within these grid boxes. Then these grid boxes are divided in
        # two triangles, which enables the use of a barycentric interpolation technique later on. The result of this
        # step is a grid with the same shape as the radar grid indicating the corresponding dem grid box and triangle
        # inside the grid box.

        triangle_coors = point_coor[delaunay.simplices]

        dem_min_line = np.ceil(np.min(triangle_coors[:, :, 0])).astype(np.int32)
        dem_max_line = np.floor(np.max(triangle_coors[:, :, 0])).astype(np.int32)
        dem_min_pixel = np.ceil(np.min(triangle_coors[:, :, 1])).astype(np.int32)
        dem_max_pixel = np.floor(np.max(triangle_coors[:, :, 1])).astype(np.int32)

        del triangle_coors

        # This is done in batches with the same maximum number of pixels inside, to prevent massive memory usage.
        max_lines = (np.ceil((dem_max_line - dem_min_line))).astype(np.int16) + 1
        m_lin = np.unique(max_lines)

        grid_triangle_id = np.zeros(shape).astype(np.int32)

        for l in m_lin:

            l_ids = np.where(max_lines == l)[0]
            max_pixels = (np.ceil((dem_max_pixel[l_ids] - dem_min_pixel[l_ids]))).astype(np.int16) + 1
            m_pix = np.unique(max_pixels)

            for p in m_pix:

                ids = l_ids[np.where(max_pixels == p)[0]]
                dem_max_num = [l, p]

                p_mesh, l_mesh = np.meshgrid(np.arange(int(np.floor(dem_max_num[1]))),
                                             np.arange(int(np.floor(dem_max_num[0]))))
                data_p = np.ravel(np.ravel(p_mesh)[None, :] + (dem_min_pixel[ids] - s_pix)[:, None] + s_pix).astype('int32')
                data_l = np.ravel(np.ravel(l_mesh)[None, :] + (dem_min_line[ids] - s_lin)[:, None] + s_lin).astype('int32')

                # get the corresponding dem ids
                triangle_id = np.ravel(ids[:, None] * np.ones(shape=(1, len(np.ravel(p_mesh))), dtype='int32')[None, :]).astype('int32')

                # From here we filter all created points using a rectangular bounding box followed by a step
                # get rid of all the points outside the bounding box
                dem_valid = ((data_p <= dem_max_pixel[triangle_id]) * (data_p >= dem_min_pixel[triangle_id]) *  # within bounding box
                             (data_l <= dem_max_line[triangle_id]) * (data_l >= dem_min_line[triangle_id]) *  # within bounding box
                             (s_lin <= data_l) * (data_l <= s_lin + shape[0]) *  # Within image
                             (s_pix <= data_p) * (data_p <= s_pix + shape[1]))
                data_p = data_p[dem_valid]
                data_l = data_l[dem_valid]
                triangle_id = triangle_id[dem_valid]

                # Now check whether the remaining points are inside the triangle using a barycentric check.
                in_triangle = Irregular2Regular.point_in_triangle(delaunay, triangle_id, point_coor, data_l, data_p)
                data_p = data_p[in_triangle]
                data_l = data_l[in_triangle]
                triangle_id = triangle_id[in_triangle]

                # Finally assign the different pixels in the new dem grid to the correct triangle in the delaunay
                # triangulation.
                grid_triangle_id[data_l, data_p] = triangle_id

        return grid_triangle_id

    @staticmethod
    def barycentric_interpolation(delaunay, triangle_id, point_coor, line_coor, pixel_coor, dem_vals):
        # This function interpolates all grid points based on the four surrounding points. We will use a barycentric
        # weighting technique. This method interpolates from an irregular to a regular grid.

        v, w = Irregular2Regular.barycentric_coordinates(delaunay, triangle_id, point_coor, pixel_coor, line_coor)
        corner_ids = delaunay.simplices[triangle_id, :, :]

        new_dem_vals = (1 - v - w) * dem_vals[corner_ids[:, 0]] + v * dem_vals[corner_ids[:, 1]] + w * dem_vals[corner_ids[:, 2]]

        return new_dem_vals

    @staticmethod
    def point_in_triangle(delaunay, triangle_id, point_coor, line_coor, pixel_coor, weights=True):
        # Check if points are inside a triangle

        v, w = Irregular2Regular.barycentric_coordinates(delaunay, triangle_id, point_coor, pixel_coor, line_coor)

        valid = ((1 - v - w) >= 0) * (v >= 0) * (w >=0) * ((1 - v - w) <= 1) * (v <= 1) * (w <= 1)

        if weights:
            return valid, v[valid], w[valid]
        else:
            return valid

    @staticmethod
    def barycentric_coordinates(delaunay, triangle_id, point_coor, pixel_coor, line_coor):
        # Checks whether a point is within a triangle.

        # Calc vectors
        corner_ids = delaunay.simplices[triangle_id, :, :]

        a = point_coor[corner_ids[:, 0]]
        v_0 = point_coor[corner_ids[:, 1]] - a
        v_1 = point_coor[corner_ids[:, 2]] - a
        v_2 = np.vstack((line_coor, pixel_coor)) - a
        a = []

        # Calculate dot products
        dot00 = np.einsum('ij,ij->j', v_0, v_0)
        dot01 = np.einsum('ij,ij->j', v_0, v_1)
        dot20 = np.einsum('ij,ij->j', v_2, v_0)
        v_0 = []
        dot11 = np.einsum('ij,ij->j', v_1, v_1)
        dot21 = np.einsum('ij,ij->j', v_2, v_1)
        v_2 = []
        v_1 = []

        # compute barycentric coordinates
        np.seterr(divide='ignore')
        denom = dot00 * dot11 - dot01 * dot01
        denom[np.isinf(denom)] = -10 ** 10
        v = (dot11 * dot20 - dot01 * dot21) / denom
        w = (dot00 * dot21 - dot01 * dot20) / denom

        return v, w
