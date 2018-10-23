
from image_data import ImageData
from find_coordinates import FindCoordinates
import os
from collections import OrderedDict, defaultdict


class Unwrap(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, meta, s_lin=0, s_pix=0, lines=0, step='interferogram', file_type='', multilook='', oversampling='', offset='', run=False):
        # There are three options for processing:
        # 1. Only give the meta_file, all other information will be read from this file. This can be a path or an
        #       ImageData object.
        # 2. Give the data files (crop, new_line, new_pixel). No need for metadata in this case
        # 3. Give the first and last line plus the buffer of the input and output

        if isinstance(meta, str):
            if len(meta) != 0:
                self.meta = ImageData(meta, 'interferogram')
        elif isinstance(meta, ImageData):
            self.meta = meta

        #
        if s_lin != 0 or s_pix != 0 or lines != 0:
            print('Unwrapping of partial ifg not supported currently.')
            return

        if file_type == '':
            file_type = step

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        self.sample, self.multilook, self.oversampling, self.offset, coor_in, coor_out = FindCoordinates.multilook_coors(ifg=self.meta, multilook=multilook, oversampling=oversampling, offset=offset)

        # Create the command to do unwrapping.
        in_file = self.meta.data_files[step][file_type + self.sample]
        self.shape = self.meta.data_sizes[step][file_type + self.sample]
        lines = str(self.shape[1])

        self.out_file = os.path.join(os.path.dirname(in_file), 'unwrapped_ifg' + self.sample + '.raw')

        conf_file = os.path.join(os.path.dirname(in_file), 'unwrap' + self.sample + '.conf')
        c = open(conf_file, 'w+')

        c.write('INFILE ' + in_file + '\n')
        c.write('LINELENGTH ' + str(lines) + '\n')

        if 'coherence' in self.meta.data_files.keys():
            if 'coherence' + self.sample in self.meta.data_files['coherence'].keys():
                c.write('CORRFILE ' + self.meta.data_files['coherence']['coherence' + self.sample] + '\n')
                c.write('CORRFILEFORMAT		FLOAT_DATA\n')

        c.write('OUTFILE ' + os.path.join(os.path.dirname(in_file), 'unwrapped_ifg' + self.sample + '.raw') + '\n')
        c.write('OUTFILEFORMAT		FLOAT_DATA\n')
        c.write('LOGFILE ' + os.path.join(os.path.dirname(in_file), 'unwrapped_ifg' + self.sample + '.log') + '\n')
        c.write('STATCOSTMODE SMOOTH')
        c.close()

        self.command = 'snaphu -f ' + conf_file
        self.step = step

    def __call__(self):
        # Do the unwrapping and create a metadata file.

        if os.path.exists(self.out_file):
            print('Unwrapping for this file is already done')
            return

        os.system(self.command)
        self.add_meta_data(self.meta, self.step, self.shape, self.sample, self.multilook, self.offset)

    @staticmethod
    def add_meta_data(meta, step):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'unwrap' in ifg.processes.keys():
            meta_info = ifg.processes['unwrap']
        else:
            meta_info = OrderedDict()

        dat = 'Data' + sample
        meta_info[dat + '_output_file'] = 'unwrapped_ifg' + sample + '.raw'
        meta_info[dat + '_output_format'] = 'real4'

        meta_info[dat + '_lines'] = str(shape[0])
        meta_info[dat + '_pixels'] = str(shape[1])
        meta_info[dat + '_first_line'] = ifg.processes[step][dat + '_first_line']
        meta_info[dat + '_first_pixel'] = ifg.processes[step][dat + '_first_pixel']
        meta_info[dat + '_multilook_azimuth'] = str(multilook[0])
        meta_info[dat + '_multilook_range'] = str(multilook[1])
        meta_info[dat + '_offset_azimuth'] = str(offset[0])
        meta_info[dat + '_offset_range'] = str(offset[1])

        ifg.image_add_processing_step('unwrap', meta_info)

    @staticmethod
    def processing_info():

        # Information on this processing step
        input_dat = defaultdict()
        input_dat['slave']['coreg'] = ['new_line', 'new_pixel']
        input_dat['slave']['resample'] = ['Data']

        output_dat = defaultdict()
        output_dat['slave']['reramp'] = ['Data']

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 5

        return input_dat, output_dat, mem_use
