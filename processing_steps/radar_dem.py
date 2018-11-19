# This function does the resampling of a radar grid based on different kernels.
# In principle this has the same functionality as some other
from image_data import ImageData
from find_coordinates import FindCoordinates
from collections import OrderedDict, defaultdict
from coordinate_system import CoordinateSystem
import numpy as np
import logging
import os
from shapely.geometry import Polygon


class RadarDem(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, coordinates, s_lin=0, s_pix=0, lines=0, coor_in=''):
        # There are three options for processing:
        # 1. Only give the meta_file, all other information will be read from this file. This can be a path or an
        #       ImageData object.
        # 2. Give the data files (crop, new_line, new_pixel). No need for metadata in this case
        # 3. Give the first and last line plus the buffer of the input and output

        if isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        # Load coordinates
        self.s_lin = s_lin
        self.s_pix = s_pix
        self.coordinates = coordinates
        self.sample = coordinates.sample
        self.first_pixel = coordinates.first_pixel
        self.first_line = coordinates.first_line
        self.multilook = coordinates.multilook
        self.shape, self.lines, self.pixels = RadarDem.find_coordinates(meta, s_lin, s_pix, lines, coordinates)

        self.used_grids = []
        self.dem_max_line = []
        self.dem_min_line = []
        self.dem_max_pixel = []
        self.dem_min_pixel = []

        # Load input data and check whether extend of input data is large enough.
        if isinstance(coor_in, CoordinateSystem):
            dem_type = coor_in.sample
        else:
            if 'import_DEM' in self.meta.processes.keys():
                dem_types = [key_str[3:-12] for key_str in self.meta.processes['import_DEM'] if
                             key_str.endswith('output_file') and not key_str.startswith('DEM_q')]
                if len(dem_types) > 0:
                    dem_type = dem_types[0]
                else:
                    print('No imported DEM found. Please rerun import_DEM function')
                    return
            else:
                print('Import a DEM first using the import_DEM function!')
                return

        self.dem, self.dem_line, self.dem_pixel = RadarDem.source_dem_extend(self.meta, self.lines, self.pixels, dem_type)

        # Initialize intermediate steps.
        self.dem_id = np.array([])
        self.first_triangle = np.array([])

        # Initialize the results
        self.radar_dem = np.zeros(self.shape)

    def __call__(self):
        if len(self.dem) == 0 or len(self.dem_line) == 0 or len(self.dem_pixel) == 0:
            print('Missing input data for creating radar DEM for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Here the actual geocoding is done.
            # First calculate the heights using an external DEM. This generates the self.height grid..
            self.dem_pixel2grid()

            # Find coordinates and matching interpolation areas
            self.radar_in_dem_grid()

            # Then do the interpolation
            self.dem_barycentric_interpolation()

            # Data can be saved using the create output files and add meta data function.
            self.add_meta_data(self.meta, self.coordinates)
            self.meta.image_new_data_memory(self.radar_dem, 'radar_DEM', self.s_lin, self.s_pix, file_type='radar_DEM' + self.sample)
            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed creating radar DEM for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed creating radar DEM for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def source_dem_extend(meta, lines, pixels, dem_type, buf=3):

        lin_key = 'DEM_line' + dem_type
        pix_key = 'DEM_pixel' + dem_type
        dem_key = 'DEM' + dem_type
        new_dat = True

        if lin_key in meta.data_memory['inverse_geocode'].keys() and \
                pix_key in meta.data_memory['inverse_geocode'].keys() and \
                dem_key in meta.data_memory['inverse_geocode'].keys():
            # If they are already loaded we check whether the extend is large enough.

            # Load the predefined first and last lines and pixels.
            pix = meta.data_memory['inverse_geocode'][pix_key]
            lin = meta.data_memory['inverse_geocode'][lin_key]

            outer_pix = np.concatenate([pix[0, :], pix[:, -1], np.flip(pix[-1, :], 0), np.flip(pix[:, 0], 0)])
            outer_lin = np.concatenate([lin[0, :], lin[:, -1], np.flip(lin[-1, :], 0), np.flip(lin[:, 0], 0)])

            convex_hull_in_dem = Polygon([[y, x] for y, x in zip(outer_lin, outer_pix)])
            convex_hull_out_dem = Polygon([[lines[0] - buf, pixels[0] - buf,   [lines[0] - buf, pixels[-1] + buf],
                                          [lines[-1] - buf, pixels[-1] + buf], [lines[-1] + buf, pixels[0] - buf]]])

            # If the input dem fully covers the output dem we do not need to load the data again
            if convex_hull_in_dem.contains(convex_hull_out_dem):
                new_dat = False

                dem_line = meta.data_memory['inverse_geocode'][lin_key]
                dem_pixel = meta.data_memory['inverse_geocode'][pix_key]
                dem = meta.data_memory['inverse_geocode'][dem_key]

        if new_dat:
            meta.read_data_memmap('inverse_geocode', lin_key)
            mem_line = meta.data_disk['inverse_geocode'][lin_key]
            meta.read_data_memmap('inverse_geocode', pix_key)
            mem_pixel = meta.data_disk['inverse_geocode'][pix_key]

            shp = mem_line.shape
            # Look for the region from the image we have to load, if not whole file is loaded in memory already.
            s_lin_region = np.max(mem_line, 1) < lines[0]
            s_pix_region = np.max(mem_pixel, 0) < pixels[0]
            e_lin_region = np.min(mem_line, 1) > lines[-1]
            e_pix_region = np.min(mem_pixel, 0) > pixels[-1]

            region_lin = np.asarray([s == e for s, e in zip(s_lin_region.ravel(), e_lin_region.ravel())])
            region_pix = np.asarray([s == e for s, e in zip(s_pix_region.ravel(), e_pix_region.ravel())])
            s_lin_source = np.maximum(0, np.min(np.argwhere(region_lin)) - buf)
            s_pix_source = np.maximum(0, np.min(np.argwhere(region_pix)) - buf)
            e_lin_source = np.minimum(shp[0], np.max(np.argwhere(region_lin)) + buf)
            e_pix_source = np.minimum(shp[1], np.max(np.argwhere(region_pix)) + buf)

            shape_source = [e_lin_source - s_lin_source, e_pix_source - s_pix_source]

            # Load the input data
            dem_line = meta.image_load_data_memory('inverse_geocode', s_lin_source, s_pix_source,
                                                             shape_source, lin_key)
            meta.clean_memory('inverse_geocode', [lin_key])
            dem_pixel = meta.image_load_data_memory('inverse_geocode', s_lin_source, s_pix_source,
                                                              shape_source, pix_key)
            dem = meta.image_load_data_memory('import_DEM', s_lin_source, s_pix_source,
                                                        shape_source, dem_key)

        return dem, dem_line, dem_pixel

    @staticmethod
    def find_coordinates(meta, s_lin, s_pix, n_lines, coordinates, ml_coors=False):

        if isinstance(coordinates, CoordinateSystem):
            if not coordinates.grid_type == 'radar_coordinates':
                print('Other grid types than radar coordinates not supported yet.')
                return
        else:
            print('coordinates should be an CoordinateSystem object')

        if len(coordinates.interval_lines) == 0:
            coordinates.add_res_info(meta)

        shape = coordinates.shape
        if n_lines != 0:
            l = np.minimum(n_lines, shape[0] - s_lin)
        else:
            l = shape[0] - s_lin
        shape = [l, shape[1] - s_pix]

        lines = coordinates.interval_lines[s_lin: s_lin + shape[0]] + coordinates.first_line
        pixels = coordinates.interval_pixels[s_pix: s_pix + shape[1]] + coordinates.first_pixel

        if ml_coors:
            ml_lines = coordinates.ml_lines[s_lin: s_lin + shape[0]] + coordinates.first_line
            ml_pixels = coordinates.ml_pixels[s_pix: s_pix + shape[1]] + coordinates.first_pixel

            return shape, lines, pixels, ml_lines, ml_pixels
        else:
            return shape, lines, pixels

    @staticmethod
    def add_meta_data(meta, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if 'radar_DEM' in meta.processes.keys():
            meta_info = meta.processes['radar_DEM']
        else:
            meta_info = OrderedDict()

        meta_info = coordinates.create_meta_data(['radar_DEM'], ['real4'], meta_info)

        meta.image_add_processing_step('radar_DEM', meta_info)

    @staticmethod
    def processing_info(coor_out, coor_in='', meta_type='cmaster'):

        # Three input files needed Dem, Dem_line and Dem_pixel
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()

        input_dat[meta_type]['import_DEM']['DEM']['file'] = ['DEM' + coor_in.sample + '.raw']
        input_dat[meta_type]['import_DEM']['DEM']['coordinates'] = coor_in
        input_dat[meta_type]['import_DEM']['DEM']['slice'] = coor_in.slice
        input_dat[meta_type]['import_DEM']['DEM']['coor_change'] = 'resample'

        for dat_type in ['DEM_pixel', 'DEM_line']:
            input_dat[meta_type]['inverse_geocode'][dat_type]['file'] = [dat_type + coor_in.sample + '.raw']
            input_dat[meta_type]['inverse_geocode'][dat_type]['coordinates'] = coor_in
            input_dat[meta_type]['inverse_geocode'][dat_type]['slice'] = coor_in.slice
            input_dat[meta_type]['inverse_geocode'][dat_type]['coor_change'] = 'resample'

        # One output file created radar dem
        output_dat = recursive_dict()
        output_dat[meta_type]['radar_DEM']['radar_DEM']['files'] = ['radar_DEM' + coor_out.sample + '.raw']
        output_dat[meta_type]['radar_DEM']['radar_DEM']['coordinate'] = coor_out
        output_dat[meta_type]['radar_DEM']['radar_DEM']['slice'] = coor_out.slice

        # Number of times input data is used in ram. Bit difficult here but 15 times is ok guess.
        mem_use = 15

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('radar_DEM', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk('radar_DEM', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('radar_DEM', file_type, coordinates)


    #######################################################################################

    # Following functions are used to do the actual computations.

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
        self.used_grids = np.where(((self.dem_max_line > self.lines[0] - self.multilook[0]) * (self.dem_min_line < self.lines[-1] + self.multilook[0]) *
                      (self.dem_max_pixel > self.pixels[0] - self.multilook[1]) * (self.dem_min_pixel < self.pixels[-1] + self.multilook[1])))[0]
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
        max_lines = (np.ceil((self.dem_max_line - self.dem_min_line) / self.multilook[0])).astype(np.int16) + 1
        m_lin = np.unique(max_lines)

        self.dem_id = np.zeros(self.shape).astype(np.int32)
        self.first_triangle = np.zeros(self.shape).astype(np.bool)

        for l in m_lin:

            l_ids = np.where(max_lines == l)[0]
            max_pixels = (np.ceil((self.dem_max_pixel[l_ids] - self.dem_min_pixel[l_ids]) / self.multilook[1])).astype(np.int16) + 1
            m_pix = np.unique(max_pixels)

            for p in m_pix:

                ids = l_ids[np.where(max_pixels == p)[0]]
                dem_max_num = [l, p]

                p_mesh, l_mesh = np.meshgrid(range(int(np.floor(dem_max_num[1]))),
                                             range(int(np.floor(dem_max_num[0]))))
                dem_p = (np.ravel(np.ravel(p_mesh)[None, :] * self.multilook[1] +
                                  np.ceil((self.dem_min_pixel[ids] - self.pixels[0]) / self.multilook[1])[:, None]
                                  * self.multilook[1] + self.pixels[0])).astype('int32')
                dem_l = (np.ravel((np.ravel(l_mesh)[None, :] * self.multilook[0]) +
                                  np.ceil((self.dem_min_line[ids] - self.lines[0]) / self.multilook[0])[:, None]
                                  * self.multilook[0] + self.lines[0])).astype('int32')

                # get the corresponding dem ids
                dem_id = np.ravel(ids[:, None] * np.ones(shape=(1, len(np.ravel(p_mesh))), dtype='int32')[None, :]
                                  ).astype('int32')

                # From here we filter all created points using a rectangular bounding box followed by a step
                # get rid of all the points outside the bounding box
                dem_valid = ((dem_p <= self.dem_max_pixel[dem_id]) * (dem_p >= self.dem_min_pixel[dem_id]) * # within bounding box
                             (dem_l <= self.dem_max_line[dem_id]) * (dem_l >= self.dem_min_line[dem_id]) *   # within bounding box
                             (self.lines[0] <= dem_l) * (dem_l <= self.lines[-1]) *            # Within image
                             (self.pixels[0] <= dem_p) * (dem_p <= self.pixels[-1]))

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
                    self.dem_id[((dem_l - self.lines[0]) / self.multilook[0]).astype(np.int32),
                                ((dem_p - self.pixels[0]) / self.multilook[1]).astype(np.int32)] = grid_id[in_gridbox]
                    self.first_triangle[((dem_l - self.lines[0]) / self.multilook[0]).astype(np.int32),
                                        ((dem_p - self.pixels[0]) / self.multilook[1]).astype(np.int32)] = first_triangle[in_gridbox]

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
