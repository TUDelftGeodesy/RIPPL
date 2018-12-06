
from image_data import ImageData
import os
from collections import OrderedDict, defaultdict
from coordinate_system import CoordinateSystem


class Unwrap(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, ifg_meta, coordinates, s_lin=0, s_pix=0, lines=0, step='interferogram', file_type='interferogram'):
        # There are three options for processing:
        # 1. Only give the meta_file, all other information will be read from this file. This can be a path or an
        #       ImageData object.
        # 2. Give the data files (crop, new_line, new_pixel). No need for metadata in this case
        # 3. Give the first and last line plus the buffer of the input and output

        if isinstance(ifg_meta, ImageData):
            self.ifg_meta = ifg_meta
        else:
            return

        if file_type == '':
            file_type = step

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        self.coordinates = coordinates
        self.sample = self.coordinates.sample

        # Create the command to do unwrapping.
        in_file = self.ifg_meta.data_files[step][file_type + self.sample]
        self.shape = self.ifg_meta.data_sizes[step][file_type + self.sample]
        lines = str(self.shape[1])

        self.out_file = os.path.join(os.path.dirname(in_file), 'unwrap' + self.sample + '.raw')

        conf_file = os.path.join(os.path.dirname(in_file), 'unwrap' + self.sample + '.conf')
        c = open(conf_file, 'w+')

        c.write('INFILE ' + in_file + '\n')
        c.write('LINELENGTH ' + str(lines) + '\n')

        if 'coherence' in self.ifg_meta.data_files.keys():
            if 'coherence' + self.sample in self.ifg_meta.data_files['coherence'].keys():
                c.write('CORRFILE ' + self.ifg_meta.data_files['coherence']['coherence' + self.sample] + '\n')
                c.write('CORRFILEFORMAT		FLOAT_DATA\n')

        c.write('OUTFILE ' + os.path.join(os.path.dirname(in_file), 'unwrap' + self.sample + '.raw') + '\n')
        c.write('OUTFILEFORMAT		FLOAT_DATA\n')
        c.write('LOGFILE ' + os.path.join(os.path.dirname(in_file), 'unwrap' + self.sample + '.log') + '\n')
        c.write('STATCOSTMODE SMOOTH')
        c.close()

        self.command = 'snaphu -f ' + conf_file
        self.step = step

    def __call__(self):
        # Do the unwrapping and create a metadata file.

        if os.path.exists(self.out_file):
            os.remove(self.out_file)
        os.system(self.command)

        # Add meta data.
        self.add_meta_data(self.ifg_meta, self.coordinates)

        return True

    @staticmethod
    def add_meta_data(ifg_meta, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'unwrap' in ifg_meta.processes.keys():
            meta_info = ifg_meta.processes['unwrap']
        else:
            meta_info = OrderedDict()

        meta_info = coordinates.create_meta_data(['unwrap'], ['real4'], meta_info)
        ifg_meta.image_add_processing_step('unwrap', meta_info)

    @staticmethod
    def processing_info(coordinates):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Three input files needed x, y, z coordinates
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        input_dat['ifg']['interferogram']['interferogram' + coordinates.sample]['file'] = 'interferogram' + coordinates.sample + '.raw'
        input_dat['ifg']['interferogram']['interferogram' + coordinates.sample]['coordinates'] = coordinates
        input_dat['ifg']['interferogram']['interferogram' + coordinates.sample]['slice'] = coordinates.slice

        # line and pixel output files.
        output_dat = recursive_dict()
        output_dat['ifg']['unwrap']['unwrap' + coordinates.sample]['file'] = 'unwrap' + coordinates.sample + '.raw'
        output_dat['ifg']['unwrap']['unwrap' + coordinates.sample]['coordinates'] = coordinates
        output_dat['ifg']['unwrap']['unwrap' + coordinates.sample]['slice'] = coordinates.slice

        # Number of times input data is used in ram. Bit difficult here but 20 times is ok guess.
        mem_use = 1

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        print('Not needed for unwrapping')

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk

        print('Not needed for unwrapping')

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk

        print('No memory data for unwrapping')
