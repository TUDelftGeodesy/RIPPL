# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np
import copy

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData

from rippl.resampling.multilook_irregular import MultilookIrregular
from rippl.resampling.multilook_regular import MultilookRegular
from rippl.resampling.coor_new_extend import CoorNewExtend
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates
from rippl.resampling.grid_transforms import GridTransforms


# noinspection PyUnboundLocalVariable
class MultilookProcess(Process):  # Change this name to the one of your processing step.

    def __call__(self, memory_in=True):
        """
        This function does basically the same as the __call__ function, but assumes that we apply no multiprocessing
        and or pipeline processing. Therefore it includes an extended number of steps:
        1. Reading in of input data
        2. Creation of output data on disk
        3. Create output memory files
        4. Perform the actual processing
        5. Save data to disk
        6. Clean memory steps.

        :return:
        """

        self.init_super()
        if self.process_finished and self.process_on_disk:
            print('Process already finished')
            return

        # Create the input and output info
        self.load_input_info()

        self.load_input_data_files()
        self.create_output_data_files()
        self.create_memory()
        self.multilook_calculations()

        self.clean_memory()
        self.out_processing_image.save_json()

        self.process_finished = True

    def __getitem__(self, key):
        # Check if there is a memory file with this name and give the output.

        if key in self.in_images.keys() or key in self.out_images.keys():
            if key in self.in_images.keys():
                get_data = self.in_images[key].disk
            else:
                get_data = self.out_images[key].disk

            dtype = get_data['meta']['dtype']
            data = get_data['data']

            if self.block_processing:
                line_no = self.no_block * self.no_lines
                data = ImageData.disk2memory(data[line_no: line_no + self.block_shape[0], :], dtype)
            elif not self.block_processing:
                data = ImageData.disk2memory(data, dtype)

        elif key in list(self.ml_out_data.keys()) and not self.block_processing:
            data = self.ml_out_data[key]
        elif (key in list(self.ml_in_data.keys()) or key in ['lines', 'pixels']) and self.block_processing:
            data = self.ml_in_data[key]

        else:
            raise LookupError('The input or output dataset ' + key + ' does not exist.')

        return data

    def __setitem__(self, key, data):
        # Set the data of one variable

        shape_in = self.coordinate_systems['in_coor'].shape

        if not self.block_processing:
            # If the shape is exactly the same as the output, we assume that we are dealing with the already multi-
            # looked image. In the very exceptional case that both are the same size, this will throw an error.
            # However, this is very unlikely so we like to use the method this way for ease of use.
            if key in list(self.out_images.keys()):
                dtype = self.out_images[key].disk['meta']['dtype']
                self.out_images[key].disk['data'] = ImageData.memory2disk(data, dtype)
            elif key in self.settings['multilooked_grids']:
                self.ml_out_data[key] = data
            else:
                raise KeyError('This output type is not defined! It should be defined in the self.setting["multilooked_grids"]')

        elif self.block_processing:
            # If the shapes are different the data is going to be multilooked.
            if key in list(self.out_images.keys()):
                if self.out_images[key].disk['data'].shape == shape_in:
                    dtype = self.out_images[key].disk['meta']['dtype']
                    line_no = self.no_block * self.no_lines
                    self.out_images[key].disk['data'][line_no: line_no + self.block_shape[0], :] = ImageData.memory2disk(data, dtype)
                elif key in self.settings['multilooked_grids']:
                    self.ml_in_data[key] = data
            elif key in self.settings['multilooked_grids'] or key in ['lines', 'pixels', 'incidence_angle']:
                self.ml_in_data[key] = data
            else:
                raise KeyError('This output type is not defined! It should be defined in the self.setting["multilooked_grids"]')

    def add_no_of_looks(self, no_of_looks=False):
        """
        This function is used to add a number of looks output to a process.

        Returns
        -------

        """

        self.settings['no_of_looks'] = no_of_looks

        # Now add to the outputs.
        if no_of_looks:
            self.output_info['file_types'].append('no_of_looks')
            self.output_info['data_types'].append('int32')

    def multilook_calculations(self):
        """
        Calculation of multilooked image using batch size.

        """

        self.block_processing = True

        # Check if the batch size variable exists.
        batch_size = getattr(self, 'batch_size', 10000000)
        regular = getattr(self, 'regular', False)

        out_shape = self.coordinate_systems['out_coor'].shape
        shape = self.coordinate_systems['in_coor'].shape
        self.no_lines = int(np.ceil(batch_size / shape[1]))
        self.no_blocks = int(np.ceil(shape[0] / self.no_lines))

        for file_type in self.settings['multilooked_grids']:
            if file_type not in self.out_images.keys():
                self.ml_out_data[file_type] = np.zeros(tuple(out_shape))

        # Check if there is a lines/pixels and the multilooking is not regular
        if not regular:
            no_input = False
            if 'no_line_pixel_input' in self.settings.keys():
                if self.settings['no_line_pixel_input'] == True:
                    no_input = True

            if 'out_irregular_grids' not in self.settings.keys() and not no_input:
                lines = self.in_images['lines'].disk['data']
                pixels = self.in_images['pixels'].disk['data']
            elif not no_input:
                lines = self.in_images[self.settings['out_irregular_grids'][0]].disk['data']
                pixels = self.in_images[self.settings['out_irregular_grids'][1]].disk['data']

        self.coordinate_systems['out_coor'].create_radar_lines()
        self.coordinate_systems['in_coor'].create_radar_lines()
        looks = np.zeros(self.coordinate_systems['out_coor'].shape)

        # Quit processing if number of pixels/line can be the same. (Very unlikely, but good to catch)
        block_shape = (self.no_lines, shape[1])
        last_shape = (shape[0] - (self.no_blocks - 1) * self.no_lines, shape[1])
        if block_shape == out_shape or last_shape == out_shape:
            raise LookupError('Shape of processing block and output multilooking image are the same. Use a different'
                              'batch size and try again!')

        for self.no_block in range(self.no_blocks):
            self.ml_coordinates = copy.deepcopy(self.coordinate_systems['in_coor'])
            self.ml_coordinates.first_line += self.no_block * self.no_lines
            self.ml_coordinates.shape[0] = np.minimum(shape[0] - self.no_block * self.no_lines, self.no_lines)
            self.block_shape = self.ml_coordinates.shape

            if not regular:
                if no_input:
                    ml_lines, ml_pixels = self.calc_line_pixel()
                else:
                    line_no = self.no_block * self.no_lines
                    ml_lines = np.copy(lines[line_no: line_no + self.block_shape[0], :])
                    ml_pixels = np.copy(pixels[line_no: line_no + self.block_shape[0], :])

            print('Processing ' + str(self.no_block) + ' out of ' + str(self.no_blocks))
            self.before_multilook_calculations()

            for file_type in self.settings['multilooked_grids']:
                if regular:
                    multilook = MultilookRegular(self.ml_coordinates, self.coordinate_systems['out_coor'])
                    multilook.apply_multilooking(self.ml_in_data[file_type])
                    self.block_processing = False
                    self[file_type] += multilook.multilooked
                    self.block_processing = True
                elif not regular:
                    multilook = MultilookIrregular(self.ml_coordinates, self.coordinate_systems['out_coor'])
                    multilook.create_conversion_grid(ml_lines, ml_pixels)
                    multilook.apply_multilooking(self.ml_in_data[file_type])
                    self.block_processing = False
                    self[file_type] += multilook.multilooked
                    self.block_processing = True
                    looks += multilook.looks

        no_types = len(list(self.settings['multilooked_grids']))
        self.block_processing = False

        if 'no_of_looks' in self.settings.keys():
            if self.settings['no_of_looks']:
                self['no_of_looks'][:, :] = looks

        for file_type in self.settings['multilooked_grids']:
            valid_pixels = ((self[file_type] != 0) * (looks !=0))
            self[file_type][valid_pixels] /= (looks[valid_pixels] / no_types)

        self.after_multilook_calculations()

    def calc_line_pixel(self):
        """
        Here the lines and pixels are calculated directly without using the prepare multilooking step.
        This is faster for smaller stacks, but slower for large stacks.

        Returns
        -------

        """

        orbit = self.processing_images['slave'].find_best_orbit('original')
        self.ml_coordinates.create_radar_lines()        # type: CoordinateSystem

        orbit_coor = OrbitCoordinates(orbit=orbit, coordinates=self.ml_coordinates)

        if self.coordinate_systems['out_coor'].grid_type == 'radar_coordinates':
            if self.settings['regular']:
                pass
            else:
                raise TypeError('Multilooking between two different radar systems is not supported.')

        if 'dem' in self.settings.keys():
            dem = self['dem']
        else:
            R, geoid_h = orbit_coor.globe_center_distance(self.ml_coordinates.center_lat, self.ml_coordinates.center_lon)
            dem = np.ones(self.ml_coordinates.shape) * geoid_h

        if self.ml_coordinates.grid_type == 'radar_coordinates':
            orbit_coor.height = dem
            orbit_coor.approx_lph2xyz()
            orbit_coor.xyz2ell()

            self['incidence_angle'] = np.reshape(orbit_coor.incidence, self.ml_coordinates.shape)
            lat = np.reshape(orbit_coor.lat, self.ml_coordinates.shape)
            lon = np.reshape(orbit_coor.lon, self.ml_coordinates.shape)
        elif self.ml_coordinates.grid_type == 'geographic':
            lat, lon = self.ml_coordinates.create_latlon_grid()
        elif self.ml_coordinates.grid_type == 'projection':
            x, y = self.ml_coordinates.create_xy_grid()
            lat, lon = self.ml_coordinates.proj2ell(x, y)

        # Final step is to calculate the lines and pixels in the final output.
        transform = GridTransforms(self.coordinate_systems['out_coor'], self.ml_coordinates)
        transform.add_dem(dem)
        transform.add_lat_lon(lat, lon)

        lines, pixels = transform()

        return lines, pixels

    def before_multilook_calculations(self):
        """
        Calculations done before multilooking

        """

        pass

    def after_multilook_calculations(self):
        """
        This function contains calculations that can be done after multilooking. Here it is empty but it can be added
        for other functions that use the multilooking.
        """

        pass

    def def_out_coor(self):
        """
        Calculate extend of output coordinates.

        :return:
        """

        new_coor = CoorNewExtend(self.coordinate_systems['in_coor'], self.coordinate_systems['out_coor'])
        self.coordinate_systems['out_coor'] = new_coor.out_coor
