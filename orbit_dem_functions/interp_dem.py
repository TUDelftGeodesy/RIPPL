# Class to create a radar dem.
from image_data import ImageData
import numpy as np
from collections import defaultdict


class InterpDem(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, s_lin=0, s_pix=0, lines=0, interval='', buffer='', buf=3, resolution='SRTM3'):
        # There are three options for processing:
        # 1. Only give the meta_file, all other information will be read from this file. This can be a path or an
        #       ImageData object.
        # 2. Give the data files (crop, new_line, new_pixel). No need for metadata in this case
        # 3. Give the first and last line plus the buffer of the input and output
        if isinstance(meta, str):
            if len(meta) != 0:
                self.meta = ImageData(meta, 'single')
        elif isinstance(meta, ImageData):
            self.meta = meta

        if not isinstance(self.meta, ImageData):
            return

        self.sample, self.interval, self.buffer, self.coors, self.in_coors, self.out_coors = self.get_interval_coors(meta, s_lin, s_pix, lines, interval, buffer)
        in_s_lin = self.in_coors[0]
        in_s_pix = self.in_coors[1]
        in_shape = self.in_coors[2]
        in_shape[0] += 1
        self.out_s_lin = self.out_coors[0]
        self.out_s_pix = self.out_coors[1]
        self.shape = self.out_coors[2]
        self.shape[0] += 1
        self.lines_out = self.coors[0]
        self.pixels_out = self.coors[1]
        self.lines = self.lines_out[self.out_s_lin:self.out_s_lin + self.shape[0]]
        self.pixels = self.pixels_out[self.out_s_pix:self.out_s_pix + self.shape[1]]
        self.first_line = self.lines[0]
        self.first_pixel = self.pixels[0]
        self.last_line = self.lines[-1]
        self.last_pixel = self.pixels[-1]

        self.resolution = resolution
        lin_key = 'Dem_line_' + self.resolution
        pix_key = 'Dem_pixel_' + self.resolution
        dem_key = 'Dem_' + self.resolution

        if lin_key not in self.meta.data_memory['inverse_geocode'].keys() or pix_key not in self.meta.data_memory['inverse_geocode'].keys():
            mem_line = self.meta.data_disk['inverse_geocode'][lin_key]
            mem_pixel = self.meta.data_disk['inverse_geocode'][pix_key]

            shp = mem_line.shape
            # Look for the region from the image we have to load, if not whole file is loaded in memory already.
            s_lin_region = np.max(mem_line, 1) < in_s_lin
            s_pix_region = np.max(mem_pixel, 0) < in_s_pix
            e_lin_region = np.min(mem_line, 1) > (in_s_lin + in_shape[0])
            e_pix_region = np.min(mem_pixel, 0) > (in_s_pix + in_shape[1])

            region_lin = np.asarray([s == e for s, e in zip(s_lin_region.ravel(), e_lin_region.ravel())])
            region_pix = np.asarray([s == e for s, e in zip(s_pix_region.ravel(), e_pix_region.ravel())])
            self.s_lin_source = np.maximum(0, np.min(np.argwhere(region_lin)) - buf)
            self.s_pix_source = np.maximum(0, np.min(np.argwhere(region_pix)) - buf)
            self.e_lin_source = np.minimum(shp[0], np.max(np.argwhere(region_lin)) + buf)
            self.e_pix_source = np.minimum(shp[1], np.max(np.argwhere(region_pix)) + buf)

            self.shape_source = [self.e_lin_source - self.s_lin_source, self.e_pix_source - self.s_pix_source]
        else:
            self.s_lin_source = self.meta.data_memory_limits['inverse_geocode'][lin_key][0]
            self.s_pix_source = self.meta.data_memory_limits['inverse_geocode'][lin_key][1]

            self.lines_source = self.meta.data_memory_sizes['inverse_geocode'][lin_key][0]
            self.pixels_source = self.meta.data_memory_sizes['inverse_geocode'][lin_key][1]
            self.shape_source = self.meta.data_memory_sizes['inverse_geocode'][lin_key]

        # Load the input data
        self.dem_line = self.meta.image_load_data_memory('inverse_geocode', self.s_lin_source, self.s_pix_source, self.shape_source, lin_key)
        self.dem_pixel = self.meta.image_load_data_memory('inverse_geocode', self.s_lin_source, self.s_pix_source, self.shape_source, pix_key)
        self.dem = self.meta.image_load_data_memory('import_dem', self.s_lin_source, self.s_pix_source,  self.shape_source, dem_key)

        self.dem_id = np.array([])
        self.first_triangle = np.array([])

        # Initialize the results
        self.radar_dem = np.zeros(self.shape)

    # The next function is used to run all interpolation steps at once.
    def interp_dem(self):
        # Find the different pixels within the dem grid
        self.dem_pixel2grid()

        # Find coordinates and matching interpolation areas
        self.radar_in_dem_grid()

        # Then do the interpolation
        self.dem_barycentric_interpolation()

    def dem_pixel2grid(self):
        # This function is used to define a grid with minimum/maximum per DEM pixel.
        # This function converts a regular grid of lines and pixels of the dem to a system of grid boxes.

        # First we calculate the pixel limits for every grid cell:
        # We use a none sparse matrix because it will become to complicated and possibly slow otherwise.
        p_num = (self.dem.shape[0] - 1) * (self.dem.shape[1] - 1)

        # Find the max and min grid values
        self.dem_max_line = - np.ones(p_num) * 10**10
        self.dem_min_line = np.ones(p_num) * 10**10
        self.dem_max_pixel = - np.ones(p_num) * 10**10
        self.dem_min_pixel = np.ones(p_num) * 10**10

        # Now find the bounding boxes
        for s_line, e_line in zip([0, 1], [-1, self.dem.shape[0]]):
            for s_pix, e_pix in zip([0, 1], [-1, self.dem.shape[1]]):
                self.dem_max_line = np.maximum(self.dem_max_line, np.ravel(self.dem_line[s_line:e_line, s_pix:e_pix]))
                self.dem_min_line = np.minimum(self.dem_min_line, np.ravel(self.dem_line[s_line:e_line, s_pix:e_pix]))
                self.dem_max_pixel = np.maximum(self.dem_max_pixel, np.ravel(self.dem_pixel[s_line:e_line, s_pix:e_pix]))
                self.dem_min_pixel = np.minimum(self.dem_min_pixel, np.ravel(self.dem_pixel[s_line:e_line, s_pix:e_pix]))

        # Finally remove grid boxes which are not used for interpolation.
        self.used_grids = np.where(((self.dem_max_line > self.first_line) * (self.dem_min_line < self.last_line) *
                      (self.dem_max_pixel > self.first_pixel) * (self.dem_min_pixel < self.last_pixel)))[0]
        self.dem_max_line = self.dem_max_line[self.used_grids]
        self.dem_min_line = self.dem_min_line[self.used_grids]
        self.dem_max_pixel = self.dem_max_pixel[self.used_grids]
        self.dem_min_pixel = self.dem_min_pixel[self.used_grids]

    def radar_in_dem_grid(self):
        # This processing_steps links the dem grid boxes to line and pixel coordinates. It returns a sparse matrix with all the
        # pixels and lines inside the grid boxes.

        # For simplicity we assume that there is minimal overlay of the points in the DEM grid. This means that
        # triangulation is already known. In fact this is also very likely for low resolution datasets like SRTM,
        # because we barely have areas steeper than the satellite look angle (50-70 degrees), perpendicular to the
        # satellite orbit.

        # For every DEM pixel we find the radar pixels within these grid boxes. Then these grid boxes are divided in
        # two triangles, which enables the use of a barycentric interpolation technique later on. The result of this
        # step is a grid with the same shape as the radar grid indicating the corresponding DEM grid box and triangle
        # inside the grid box.

        # This is done in batches with the same maximum number of pixels inside, to prevent massive memory usage.
        max_lines = (np.floor(self.dem_max_line) - np.floor(self.dem_min_line)).astype(np.int16)
        m_lin = np.unique(max_lines)

        self.dem_id = np.zeros(self.shape).astype(np.int32)
        self.first_triangle = np.zeros(self.shape).astype(np.bool)

        for l in m_lin:

            l_ids = np.where(max_lines == l)[0]
            max_pixels = (np.floor(self.dem_max_pixel[l_ids]) - np.floor(self.dem_min_pixel[l_ids])).astype(np.int16)
            m_pix = np.unique(max_pixels)

            for p in m_pix:

                ids = l_ids[np.where(max_pixels == p)[0]]
                dem_max_num = [l, p]

                p_mesh, l_mesh = np.meshgrid(range(int(np.floor(dem_max_num[1] / self.interval[1] + 1))),
                                             range(int(np.floor(dem_max_num[0] / self.interval[0] + 1))))
                dem_p = (np.ravel(np.ravel(p_mesh)[None, :] * self.interval[1] +
                                  np.ceil((self.dem_min_pixel[ids] - self.first_pixel) / self.interval[1])[:, None]
                                  * self.interval[1] + self.first_pixel)).astype('int32')
                dem_l = (np.ravel((np.ravel(l_mesh)[None, :] * self.interval[0]) +
                                  np.ceil((self.dem_min_line[ids] - self.first_line) / self.interval[0])[:, None]
                                  * self.interval[0] + self.first_line)).astype('int32')

                # get the corresponding dem ids
                dem_id = np.ravel(ids[:, None] * np.ones(shape=(1, len(np.ravel(p_mesh))), dtype='int32')[None, :]
                                  ).astype('int32')

                # From here we filter all created points using a rectangular bounding box followed by a step
                # get rid of all the points outside the bounding box
                dem_valid = ((dem_p <= self.dem_max_pixel[dem_id]) * (dem_p >= self.dem_min_pixel[dem_id]) * # within bounding box
                             (dem_l <= self.dem_max_line[dem_id]) * (dem_l >= self.dem_min_line[dem_id]) *   # within bounding box
                             (self.first_line <= dem_l) * (dem_l <= self.last_line) *            # Within image
                             (self.first_pixel <= dem_p) * (dem_p <= self.last_pixel))

                dem_p = dem_p[dem_valid]
                dem_l = dem_l[dem_valid]
                dem_id = dem_id[dem_valid]

                # Now check whether the remaining points are inside our quadrilateral
                # This is done using a double barycentric approach.
                # The result also gives us a weighting based on the distance of pixel in range and azimuth
                grid_id = self.used_grids[dem_id]
                in_gridbox, first_triangle = self.barycentric_check(self.dem_line, self.dem_pixel, grid_id, dem_l, dem_p)
                dem_p = dem_p[in_gridbox]
                dem_l = dem_l[in_gridbox]

                # The evaluated point are now added to the regular output grid, which gives us the following information
                # for later processing:
                # 1. In which dem grid cell does it fall
                # 2. Is it included in the upper left of lower right part of this gridbox.
                if len(in_gridbox) > 0:
                    self.dem_id[((dem_l - self.lines[0]) / self.interval[0]),
                                ((dem_p - self.pixels[0]) / self.interval[1])] = grid_id[in_gridbox]
                    self.first_triangle[((dem_l - self.lines[0]) / self.interval[0]),
                                        ((dem_p - self.pixels[0]) / self.interval[1])] = first_triangle[in_gridbox]

        # Finally remove the data needed to find the corresponding points in our grid.
        del self.dem_max_line, self.dem_min_line, self.dem_min_pixel, self.dem_max_pixel
        del self.used_grids

    def dem_barycentric_interpolation(self):
        # This function interpolates all grid points based on the four surrounding points. We will use a barycentric
        # weighting technique. This method interpolates from an irregular to a regular grid.

        h = np.zeros(shape=self.radar_dem.shape, dtype='float32')
        area = np.zeros(shape=self.radar_dem.shape, dtype='float32')

        # lower triangle
        dem_l, dem_p = np.where(self.first_triangle)
        l_id = self.dem_id[self.first_triangle] / (self.dem_line.shape[1] - 1)
        p_id = self.dem_id[self.first_triangle] - (l_id * (self.dem_line.shape[1] - 1))

        s_p = [0, 1, 1, 0]
        s_l = [0, 0, 1, 1]

        for c in [[3, 0, 1], [0, 1, 3], [1, 3, 0]]:  # coordinate values (ul, ur, lr, ll)
            # Calculate area of triangle
            ax = (dem_l * (self.dem_pixel[l_id + s_l[c[0]], p_id + s_p[c[0]]] -
                           self.dem_pixel[l_id + s_l[c[1]], p_id + s_p[c[1]]]) +
                  (self.dem_line[l_id + s_l[c[0]], p_id + s_p[c[0]]] *
                  (self.dem_pixel[l_id + s_l[c[1]], p_id + s_p[c[1]]] - dem_p)) +
                  (self.dem_line[l_id + s_l[c[1]], p_id + s_p[c[1]]] *
                  (dem_p - self.dem_pixel[l_id + s_l[c[0]], p_id + s_p[c[0]]])))

            # dem values are shifted to find ul,ur,lr,ll values
            area[dem_l, dem_p] += np.abs(ax)
            h[dem_l, dem_p] += np.abs(ax) * self.dem[l_id + s_l[c[2]], p_id + s_p[c[2]]]

        # upper triangle
        dem_l, dem_p = np.where(~self.first_triangle)
        l_id = self.dem_id[~self.first_triangle] / (self.dem_line.shape[1] - 1)
        p_id = self.dem_id[~self.first_triangle] - (l_id * (self.dem_line.shape[1] - 1))

        for c in [[1, 3, 2], [3, 2, 1], [2, 1, 3]]:  # coordinate values (ul, ur, lr, ll)
            # Calculate area of triangle
            ax = (dem_l * (self.dem_pixel[l_id + s_l[c[0]], p_id + s_p[c[0]]] -
                           self.dem_pixel[l_id + s_l[c[1]], p_id + s_p[c[1]]]) +
                  (self.dem_line[l_id + s_l[c[0]], p_id + s_p[c[0]]] *
                  (self.dem_pixel[l_id + s_l[c[1]], p_id + s_p[c[1]]] - dem_p)) +
                  (self.dem_line[l_id + s_l[c[1]], p_id + s_p[c[1]]] *
                  (dem_p - self.dem_pixel[l_id + s_l[c[0]], p_id + s_p[c[0]]])))

            # dem values are shifted to find ul,ur,lr,ll values
            area[dem_l, dem_p] += np.abs(ax)
            h[dem_l, dem_p] += np.abs(ax) * self.dem[l_id + s_l[c[2]], p_id + s_p[c[2]]]

        del dem_l, dem_p, l_id, p_id

        self.radar_dem = h / area

    @staticmethod
    def barycentric_check(dem_lines, dem_pixels, dem_id, dem_l, dem_p):
        # Checks whether a point is within a triangle. (combination of two triangles can be used for a quadrilateral)

        valid_id = np.zeros(dem_id.shape, dtype=bool)
        triangle = np.zeros(dem_id.shape, dtype=bool)
        l_id = dem_id / (dem_pixels.shape[1] - 1)
        p_id = dem_id - (l_id * (dem_lines.shape[1] - 1))

        s_p = [0, 1, 1, 0]
        s_l = [0, 0, 1, 1]

        for c in [2, 0]:
            # Calc vectors
            a = np.vstack((dem_lines[l_id + s_l[c], p_id + s_p[c]], dem_pixels[l_id + s_l[c], p_id + s_p[c]]))
            v_0 = np.vstack((dem_lines[l_id + 0, p_id + 1], dem_pixels[l_id + 0, p_id + 1])) - a
            v_1 = np.vstack((dem_lines[l_id + 1, p_id + 0], dem_pixels[l_id + 1, p_id + 0])) - a
            v_2 = np.vstack((dem_l, dem_p)) - a
            del a

            # Calculate dot products
            dot00 = np.einsum('ij,ij->j', v_0, v_0)
            dot01 = np.einsum('ij,ij->j', v_0, v_1)
            dot02 = np.einsum('ij,ij->j', v_0, v_2)
            del v_0
            dot11 = np.einsum('ij,ij->j', v_1, v_1)
            dot12 = np.einsum('ij,ij->j', v_1, v_2)
            del v_2, v_1

            # compute barycentric coordinates
            np.seterr(divide='ignore')
            inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
            inv_denom[np.isinf(inv_denom)] = -10**10
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom
            del inv_denom

            # Check if pixel is in triangle. Only if the point lies on one of the lines it is valid.
            valid_id[((u > 0) * (v >= 0) * (u + v < 1))] = True
            if c == 0:
                triangle[((u > 0) * (v >= 0) * (u + v < 1))] = True

        return valid_id, triangle
