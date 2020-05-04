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


class MultilookProcess(Process):  # Change this name to the one of your processing step.

    def __getitem__(self, key):
        # Check if there is a memory file with this name and give the output.
        shape_in = self.coordinate_systems['in_coor'].shape

        if key in self.in_images.keys() or key in self.out_images.keys():
            if key in self.in_images.keys():
                get_data = self.in_images[key].disk
            else:
                get_data = self.out_images[key].disk

            dtype = get_data['meta']['dtype']
            data = get_data['data']

            if tuple(get_data['data'].shape) == tuple(shape_in):
                line_no = self.no_block * self.no_lines
                data = ImageData.disk2memory(data[line_no: line_no + self.block_shape[0], :], dtype)
            else:
                data = ImageData.disk2memory(data, dtype)

        elif key in list(self.ml_out_data.keys()):
            data = self.ml_out_data[key]
        elif key in list(self.ml_in_data.keys()):
            data = self.ml_in_data[key]

        else:
            raise LookupError('The input or output dataset ' + key + ' does not exist.')

        return data

    def __setitem__(self, key, data):
        # Set the data of one variable
        shape_out = self.coordinate_systems['out_coor'].shape
        shape_in = self.coordinate_systems['in_coor'].shape

        if data.shape == tuple(shape_out):
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

        elif data.shape[0] == self.block_shape[0]:
            # If the shapes are different the data is going to be multilooked.
            if key in list(self.out_images.keys()):
                if self.out_images[key].disk['data'].shape == shape_in:
                    dtype = self.out_images[key].disk['meta']['dtype']
                    line_no = self.no_block * self.no_lines
                    self.out_images[key].disk['data'][line_no: line_no + self.block_shape[0], :] = ImageData.memory2disk(data, dtype)
                elif key in self.settings['multilooked_grids']:
                    self.ml_in_data[key] = data
            elif key in self.settings['multilooked_grids']:
                self.ml_in_data[key] = data
            else:
                raise KeyError('This output type is not defined! It should be defined in the self.setting["multilooked_grids"]')

        else:
            raise TypeError('Not able to save this type of data')

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
            if ('lines' not in self.in_images.keys() or 'pixels' not in self.in_images.keys()) \
                    and not 'no_pre_calculated_coordinates' in self.settings.keys():
                raise LookupError('There should be a lines and pixels input variable to do irregular multilooking! '
                                  'If you do a regular multilook define self.regular = True in the __init__')
            else:
                if 'out_irregular_grids' not in self.settings.keys():
                    lines = self.in_images['lines'].disk['data']
                    pixels = self.in_images['pixels'].disk['data']
                else:
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
            coordinates = copy.deepcopy(self.coordinate_systems['in_coor'])
            coordinates.first_line += self.no_block * self.no_lines
            coordinates.shape[0] = np.minimum(shape[0] - self.no_block * self.no_lines, self.no_lines)
            self.block_shape = coordinates.shape

            print('Processing ' + str(self.no_block) + ' out of ' + str(self.no_blocks))
            self.before_multilook_calculations()

            for file_type in self.settings['multilooked_grids']:
                if regular:
                    multilook = MultilookRegular(coordinates, self.coordinate_systems['out_coor'])
                    self[file_type] += multilook(self.ml_in_data[file_type])
                elif not regular:
                    multilook = MultilookIrregular(coordinates, self.coordinate_systems['out_coor'])
                    line_no = self.no_block * self.no_lines
                    if 'no_pre_calculated_coordinates' in self.settings.keys():
                        multilook.create_conversion_grid(self['lines'], self['pixels'])
                    else:
                        multilook.create_conversion_grid(
                            np.copy(lines[line_no: line_no + self.block_shape[0], :]),
                            np.copy(pixels[line_no: line_no + self.block_shape[0], :]))
                    multilook.apply_multilooking(self.ml_in_data[file_type])
                    self[file_type] += multilook.multilooked
                    looks += multilook.looks

        no_types = len(list(self.settings['multilooked_grids']))

        for file_type in self.ml_in_data.keys():
            valid_pixels = ((self[file_type] != 0) * (looks !=0))
            self[file_type][valid_pixels] /= (looks[valid_pixels] / no_types)

        self.block_processing = False
        self.after_multilook_calculations()

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
