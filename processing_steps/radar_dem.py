# This function does the resampling of a radar grid based on different kernels.
# In principle this has the same functionality as some other
from rippl.image_data import ImageData
from collections import OrderedDict, defaultdict
from rippl.coordinate_system import CoordinateSystem
import numpy as np
import logging
import os
from shapely.geometry import Polygon
import copy


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

        self.lines = self.lines[:, 0]
        self.pixels = self.pixels[0, :]
        self.dem, self.dem_line, self.dem_pixel = RadarDem.source_dem_extend(self.meta, self.lines,
                                                                             self.pixels, dem_type)
        self.no0 = (self.dem_line != 0) * (self.dem_pixel != 0)
        if self.coordinates.mask_grid:
            mask = self.meta.image_load_data_memory('create_sparse_grid', s_lin, 0, self.shape,
                                                       'mask' + self.coordinates.sample)
            self.no0 *= mask
            # TODO Implement masking for calculating radar DEM.

        self.radar_dem = []

    def __call__(self):
        if len(self.dem) == 0 or len(self.dem_line) == 0 or len(self.dem_pixel) == 0:
            print('Missing input data for creating radar DEM for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Here the actual geocoding is done.
            # First calculate the heights using an external DEM. This generates the self.height grid..
            used_grids, grid_extends = self.dem_pixel2grid(self.dem, self.dem_line, self.dem_pixel, self.lines, self.pixels, self.multilook)

            # Find coordinates and matching interpolation areas
            dem_id, first_triangle = self.radar_in_dem_grid(used_grids, grid_extends, self.lines, self.pixels, self.multilook, self.shape, self.dem_line, self.dem_pixel)
            del used_grids, grid_extends

            # Then do the interpolation
            self.radar_dem = self.dem_barycentric_interpolation(self.dem, self.dem_line, self.dem_pixel, self.shape, first_triangle, dem_id)
            del self.dem, self.dem_line, self.dem_pixel

            # Data can be saved using the create output files and add meta data function.
            self.add_meta_data(self.meta, self.coordinates)
            self.meta.image_new_data_memory(self.radar_dem, 'radar_DEM', self.s_lin, self.s_pix, file_type='DEM' + self.sample)
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

            max_lin = np.max(lines)
            min_lin = np.min(lines)
            max_pix = np.max(pixels)
            min_pix = np.min(pixels)
            convex_hull_in_dem = Polygon([[y, x] for y, x in zip(outer_lin, outer_pix)])
            convex_hull_out_dem = Polygon([[min_lin - buf, min_pix - buf,   [min_lin - buf, max_pix + buf],
                                          [max_lin - buf, max_pix + buf], [max_lin + buf, min_pix - buf]]])

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

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')
            return

        if len(coordinates.interval_lines) == 0 and coordinates.grid_type == 'radar_coordinates':
            coordinates.add_res_info(meta)
        shape = copy.copy(coordinates.shape)
        if n_lines != 0:
            l = np.minimum(n_lines, shape[0] - s_lin)
        else:
            l = shape[0] - s_lin
        shape = [l, shape[1] - s_pix]

        if coordinates.grid_type == 'radar_coordinates':
            if coordinates.sparse_grid:
                pixels = meta.image_load_data_memory('create_sparse_grid', s_lin, s_pix, shape, 'pixel' + coordinates.sample)
                lines = meta.image_load_data_memory('create_sparse_grid', s_lin, s_pix, shape, 'line' + coordinates.sample)
            else:
                line = coordinates.interval_lines[s_lin: s_lin + shape[0]] + coordinates.first_line
                pixel = coordinates.interval_pixels[s_pix: s_pix + shape[1]] + coordinates.first_pixel
                pixels, lines = np.meshgrid(pixel, line)

        else:
            lines = meta.image_load_data_memory('coor_geocode', s_lin, s_pix, shape, 'line' + coordinates.sample)
            pixels = meta.image_load_data_memory('coor_geocode', s_lin, s_pix, shape, 'pixel' + coordinates.sample)

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

        meta_info = coordinates.create_meta_data(['DEM'], ['real4'], meta_info)

        meta.image_add_processing_step('radar_DEM', meta_info)

    @staticmethod
    def processing_info(coor_out, coor_in='', meta_type='cmaster'):

        # Three input files needed Dem, Dem_line and Dem_pixel
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()

        input_dat[meta_type]['import_DEM']['DEM' + coor_in.sample]['file'] = 'DEM' + coor_in.sample + '.raw'
        input_dat[meta_type]['import_DEM']['DEM' + coor_in.sample]['coordinates'] = coor_in
        input_dat[meta_type]['import_DEM']['DEM' + coor_in.sample]['slice'] = coor_in.slice
        input_dat[meta_type]['import_DEM']['DEM' + coor_in.sample]['coor_change'] = 'resample'

        for dat_type in ['DEM_pixel', 'DEM_line']:
            input_dat[meta_type]['inverse_geocode'][dat_type + coor_in.sample]['file'] = dat_type + coor_in.sample + '.raw'
            input_dat[meta_type]['inverse_geocode'][dat_type + coor_in.sample]['coordinates'] = coor_in
            input_dat[meta_type]['inverse_geocode'][dat_type + coor_in.sample]['slice'] = coor_in.slice
            input_dat[meta_type]['inverse_geocode'][dat_type + coor_in.sample]['coor_change'] = 'resample'

        # One output file created radar dem
        output_dat = recursive_dict()
        output_dat[meta_type]['radar_DEM']['DEM' + coor_out.sample]['files'] = 'DEM' + coor_out.sample + '.raw'
        output_dat[meta_type]['radar_DEM']['DEM' + coor_out.sample]['coordinate'] = coor_out
        output_dat[meta_type]['radar_DEM']['DEM' + coor_out.sample]['slice'] = coor_out.slice

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
    @staticmethod
    def dem_pixel2grid(dem, dem_line, dem_pixel, lines, pixels, multilook):
        # This function is used to define a grid with minimum/maximum per DEM pixel.
        # This function converts a regular grid of lines and pixels of the dem to a system of grid boxes.

        # First we calculate the pixel limits for every grid cell:
        # We use a none sparse matrix because it will become to complicated and possibly slow otherwise.

        # First step is to remove all the grid cells that do not contain points (Case where DEM grid is finer than the

        # Find the max and min grid values
        used_grids = np.ones(dem.shape - np.array([1, 1])).astype(np.bool)
        for s_line, e_line in zip([0, 1], [-1, dem.shape[0]]):
            for s_pix, e_pix in zip([0, 1], [-1, dem.shape[1]]):
                used_grids *= (dem_line[s_line:e_line, s_pix:e_pix] != 0) * (dem_pixel[s_line:e_line, s_pix:e_pix] != 0)
        used_grids = np.ravel(used_grids)

        dem_max_line = - np.ones(np.sum(used_grids)) * 10**10
        dem_min_line = np.ones(np.sum(used_grids)) * 10**10

        # Now find the bounding boxes
        for s_line, e_line in zip([0, 1], [-1, dem.shape[0]]):
            for s_pix, e_pix in zip([0, 1], [-1, dem.shape[1]]):
                dem_max_line = np.maximum(dem_max_line, np.ravel(dem_line[s_line:e_line, s_pix:e_pix])[used_grids])
                dem_min_line = np.minimum(dem_min_line, np.ravel(dem_line[s_line:e_line, s_pix:e_pix])[used_grids])

        used_grids[used_grids] = (np.floor((dem_max_line - lines[0]) / multilook[0]) - np.floor((dem_min_line - lines[0]) / multilook[0])).astype(np.int16) > 0
        dem_max_pixel = - np.ones(np.sum(used_grids)) * 10**10
        dem_min_pixel = np.ones(np.sum(used_grids)) * 10**10

        for s_line, e_line in zip([0, 1], [-1, dem.shape[0]]):
            for s_pix, e_pix in zip([0, 1], [-1, dem.shape[1]]):
                dem_max_pixel = np.maximum(dem_max_pixel, np.ravel(dem_pixel[s_line:e_line, s_pix:e_pix])[used_grids])
                dem_min_pixel = np.minimum(dem_min_pixel, np.ravel(dem_pixel[s_line:e_line, s_pix:e_pix])[used_grids])

        used_grids[used_grids] = (np.floor((dem_max_pixel - pixels[0]) / multilook[1]) - np.floor((dem_min_pixel - pixels[0]) / multilook[1])).astype(np.int16) > 0
        no_grid_cells = np.sum(used_grids)

        dem_max_line = - np.ones(no_grid_cells) * 10**10
        dem_min_line = np.ones(no_grid_cells) * 10**10
        dem_max_pixel = - np.ones(no_grid_cells) * 10**10
        dem_min_pixel = np.ones(no_grid_cells) * 10**10

        for s_line, e_line in zip([0, 1], [-1, dem.shape[0]]):
            for s_pix, e_pix in zip([0, 1], [-1, dem.shape[1]]):
                dem_max_line = np.maximum(dem_max_line, np.ravel(dem_line[s_line:e_line, s_pix:e_pix])[used_grids])
                dem_min_line = np.minimum(dem_min_line, np.ravel(dem_line[s_line:e_line, s_pix:e_pix])[used_grids])
                dem_max_pixel = np.maximum(dem_max_pixel, np.ravel(dem_pixel[s_line:e_line, s_pix:e_pix])[used_grids])
                dem_min_pixel = np.minimum(dem_min_pixel, np.ravel(dem_pixel[s_line:e_line, s_pix:e_pix])[used_grids])

        # Finally remove grid boxes which are not used for interpolation.
        used_dem_grids = ((dem_max_line > lines[0] - multilook[0]) * (dem_min_line < lines[-1] + multilook[0]) *
                      (dem_max_pixel > pixels[0] - multilook[1]) * (dem_min_pixel < pixels[-1] + multilook[1]))

        used_grids[used_grids] = used_dem_grids
        used_grids = np.where(used_grids)[0]
        dem_max_line = dem_max_line[used_dem_grids]
        dem_min_line = dem_min_line[used_dem_grids]
        dem_max_pixel = dem_max_pixel[used_dem_grids]
        dem_min_pixel = dem_min_pixel[used_dem_grids]

        return used_grids, [dem_min_line, dem_max_line, dem_min_pixel, dem_max_pixel]

    @staticmethod
    def radar_in_dem_grid(used_grids, grid_extends, lines, pixels, multilook, size, dem_line, dem_pixel):
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

        dem_min_line = grid_extends[0]
        dem_max_line = grid_extends[1]
        dem_min_pixel = grid_extends[2]
        dem_max_pixel = grid_extends[3]

        # This is done in batches with the same maximum number of pixels inside, to prevent massive memory usage.
        max_lines = (np.ceil((dem_max_line - dem_min_line) / multilook[0])).astype(np.int16) + 1
        m_lin = np.unique(max_lines)

        grid_dem_id = np.zeros(size).astype(np.int32)
        grid_first_triangle = np.zeros(size).astype(np.bool)

        for l in m_lin:

            l_ids = np.where(max_lines == l)[0]
            max_pixels = (np.ceil((dem_max_pixel[l_ids] - dem_min_pixel[l_ids]) / multilook[1])).astype(np.int16) + 1
            m_pix = np.unique(max_pixels)

            for p in m_pix:

                ids = l_ids[np.where(max_pixels == p)[0]]
                dem_max_num = [l, p]

                p_mesh, l_mesh = np.meshgrid(np.arange(int(np.floor(dem_max_num[1]))),
                                             np.arange(int(np.floor(dem_max_num[0]))))
                dem_p = (np.ravel(np.ravel(p_mesh)[None, :] * multilook[1] +
                                  np.ceil((dem_min_pixel[ids] - pixels[0]) / multilook[1])[:, None]
                                  * multilook[1] + pixels[0])).astype('int32')
                dem_l = (np.ravel((np.ravel(l_mesh)[None, :] * multilook[0]) +
                                  np.ceil((dem_min_line[ids] - lines[0]) / multilook[0])[:, None]
                                  * multilook[0] + lines[0])).astype('int32')

                # get the corresponding dem ids
                dem_id = np.ravel(ids[:, None] * np.ones(shape=(1, len(np.ravel(p_mesh))), dtype='int32')[None, :]
                                  ).astype('int32')

                # From here we filter all created points using a rectangular bounding box followed by a step
                # get rid of all the points outside the bounding box
                dem_valid = ((dem_p <= dem_max_pixel[dem_id]) * (dem_p >= dem_min_pixel[dem_id]) * # within bounding box
                             (dem_l <= dem_max_line[dem_id]) * (dem_l >= dem_min_line[dem_id]) *   # within bounding box
                             (lines[0] <= dem_l) * (dem_l <= lines[-1]) *            # Within image
                             (pixels[0] <= dem_p) * (dem_p <= pixels[-1]))

                dem_p = dem_p[dem_valid]
                dem_l = dem_l[dem_valid]
                dem_id = dem_id[dem_valid]

                # Now check whether the remaining points are inside our quadrilateral
                # This is done using a double barycentric approach.
                # The result also gives us a weighting based on the distance of pixel in range and azimuth
                grid_id = used_grids[dem_id].astype(np.int32)
                in_gridbox, first_triangle = RadarDem.barycentric_check(dem_line, dem_pixel, grid_id, dem_l, dem_p)
                dem_p = dem_p[in_gridbox]
                dem_l = dem_l[in_gridbox]

                # The evaluated point are now added to the regular output grid, which gives us the following information
                # for later processing:
                # 1. In which dem grid cell does it fall
                # 2. Is it included in the upper left of lower right part of this gridbox.
                if len(in_gridbox) > 0:
                    grid_dem_id[((dem_l - lines[0]) / multilook[0]).astype(np.int32),
                                ((dem_p - pixels[0]) / multilook[1]).astype(np.int32)] = grid_id[in_gridbox]
                    grid_first_triangle[((dem_l - lines[0]) / multilook[0]).astype(np.int32),
                                        ((dem_p - pixels[0]) / multilook[1]).astype(np.int32)] = first_triangle[in_gridbox]

        return grid_dem_id, grid_first_triangle

    @staticmethod
    def dem_barycentric_interpolation(dem, dem_line, dem_pixel, size, first_triangle, dem_id):
        # This function interpolates all grid points based on the four surrounding points. We will use a barycentric
        # weighting technique. This method interpolates from an irregular to a regular grid.

        h = np.zeros(shape=size, dtype='float32')
        area = np.zeros(shape=size, dtype='float32')

        # lower triangle
        dem_l, dem_p = np.where(first_triangle)
        l_id = dem_id[first_triangle] // (dem_line.shape[1] - 1)
        p_id = dem_id[first_triangle] - (l_id * (dem_line.shape[1] - 1))

        s_p = [0, 1, 1, 0]
        s_l = [0, 0, 1, 1]

        for c in [[3, 0, 1], [0, 1, 3], [1, 3, 0]]:  # coordinate values (ul, ur, lr, ll)
            # Calculate area of triangle
            ax = (dem_l * (dem_pixel[l_id + s_l[c[0]], p_id + s_p[c[0]]] -
                           dem_pixel[l_id + s_l[c[1]], p_id + s_p[c[1]]]) +
                  (dem_line[l_id + s_l[c[0]], p_id + s_p[c[0]]] *
                  (dem_pixel[l_id + s_l[c[1]], p_id + s_p[c[1]]] - dem_p)) +
                  (dem_line[l_id + s_l[c[1]], p_id + s_p[c[1]]] *
                  (dem_p - dem_pixel[l_id + s_l[c[0]], p_id + s_p[c[0]]])))

            # dem values are shifted to find ul,ur,lr,ll values
            area[dem_l, dem_p] += np.abs(ax)
            h[dem_l, dem_p] += np.abs(ax) * dem[l_id + s_l[c[2]], p_id + s_p[c[2]]]

        # upper triangle
        dem_l, dem_p = np.where(~first_triangle)
        l_id = dem_id[~first_triangle] // (dem_line.shape[1] - 1)
        p_id = dem_id[~first_triangle] - (l_id * (dem_line.shape[1] - 1))

        for c in [[1, 3, 2], [3, 2, 1], [2, 1, 3]]:  # coordinate values (ul, ur, lr, ll)
            # Calculate area of triangle
            ax = (dem_l * (dem_pixel[l_id + s_l[c[0]], p_id + s_p[c[0]]] -
                           dem_pixel[l_id + s_l[c[1]], p_id + s_p[c[1]]]) +
                  (dem_line[l_id + s_l[c[0]], p_id + s_p[c[0]]] *
                  (dem_pixel[l_id + s_l[c[1]], p_id + s_p[c[1]]] - dem_p)) +
                  (dem_line[l_id + s_l[c[1]], p_id + s_p[c[1]]] *
                  (dem_p - dem_pixel[l_id + s_l[c[0]], p_id + s_p[c[0]]])))

            # dem values are shifted to find ul,ur,lr,ll values
            area[dem_l, dem_p] += np.abs(ax)
            h[dem_l, dem_p] += np.abs(ax) * dem[l_id + s_l[c[2]], p_id + s_p[c[2]]]

        radar_dem = h / area

        return radar_dem

    @staticmethod
    def barycentric_check(dem_lines, dem_pixels, dem_id, dem_l, dem_p):
        # Checks whether a point is within a triangle. (combination of two triangles can be used for a quadrilateral)

        valid_id = np.zeros(dem_id.shape, dtype=bool)
        triangle = np.zeros(dem_id.shape, dtype=bool)
        l_id = dem_id // (dem_pixels.shape[1] - 1)
        p_id = dem_id - (l_id * (dem_lines.shape[1] - 1))

        s_p = [0, 1, 1, 0]
        s_l = [0, 0, 1, 1]

        for c in [2, 0]:
            # Calc vectors
            a = np.vstack((dem_lines[l_id + s_l[c], p_id + s_p[c]], dem_pixels[l_id + s_l[c], p_id + s_p[c]]))
            v_0 = np.vstack((dem_lines[l_id + 0, p_id + 1], dem_pixels[l_id + 0, p_id + 1])) - a
            v_1 = np.vstack((dem_lines[l_id + 1, p_id + 0], dem_pixels[l_id + 1, p_id + 0])) - a
            v_2 = np.vstack((dem_l, dem_p)) - a
            a = []

            # Calculate dot products
            dot00 = np.einsum('ij,ij->j', v_0, v_0)
            dot01 = np.einsum('ij,ij->j', v_0, v_1)
            dot02 = np.einsum('ij,ij->j', v_0, v_2)
            v_0 = []
            dot11 = np.einsum('ij,ij->j', v_1, v_1)
            dot12 = np.einsum('ij,ij->j', v_1, v_2)
            v_2 = []
            v_1 = []

            # compute barycentric coordinates
            np.seterr(divide='ignore')
            inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
            inv_denom[np.isinf(inv_denom)] = -10**10
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom
            inv_denom = []

            # Check if pixel is in triangle. Only if the point lies on one of the lines it is valid.
            valid_id[((u > 0) * (v >= 0) * (u + v < 1))] = True
            if c == 0:
                triangle[((u > 0) * (v >= 0) * (u + v < 1))] = True

        return valid_id, triangle
