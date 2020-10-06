# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import os
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.user_settings import UserSettings


class Unwrap(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', out_coor=[], ifg='ifg', overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs
        :param CoordinateSystem out_coor: Coordinate system of the input grids.
        :param ImageProcessingData ifg: Interferogram of a master/slave combination
        :param bool overwrite: Do we overwrite a file or not?
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'unwrap'
        self.output_info['image_type'] = 'ifg'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['unwrapped']
        self.output_info['data_types'] = ['real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['ifg', 'ifg']
        self.input_info['process_types'] = ['coherence', 'interferogram']
        self.input_info['file_types'] = ['coherence', 'interferogram']
        self.input_info['polarisations'] = [polarisation, polarisation]
        self.input_info['data_ids'] = [data_id, data_id]
        self.input_info['coor_types'] = ['out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '']
        self.input_info['type_names'] = ['coherence', 'interferogram']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['ifg'] = ifg

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        settings = UserSettings()
        settings.load_settings()
        self.settings = dict()
        self.settings['snaphu_path'] = settings.snaphu_path

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(Unwrap, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        This step contains all the calculations that are done for this step. This is therefore the step that is most
        specific for every process definition.
        To run the full processing you have to initialize (__init__) and call (__call__) the object.

        To access the input data and save the output data you can the self.images dictionary. This contains all the
        memory data files. The keys are the same as the file_types and in_file_types.

        :return:
        """

        ifg_file_name = self.in_images['interferogram'].disk['meta']['file_name']
        coh_file_name = self.in_images['coherence'].disk['meta']['file_name']
        folder = self.in_images['coherence'].folder

        shape = self['interferogram'].shape
        pixels = str(shape[1])

        unwrap_name = 'unwrapped' + coh_file_name[len('coherence'):][:-4]
        out_file = os.path.join(folder, unwrap_name + '.data')
        conf_file = os.path.join(folder, unwrap_name + '.conf')

        with open(conf_file, 'w+') as c:
            c.write('INFILE ' + os.path.join(folder, ifg_file_name) + '\n')
            c.write('LINELENGTH ' + str(pixels) + '\n')

            # Add coherence file
            c.write('CORRFILE ' + os.path.join(folder, coh_file_name) + '\n')
            c.write('CORRFILEFORMAT		FLOAT_DATA\n')
            c.write('OUTFILE ' + os.path.join(folder, out_file) + '\n')
            c.write('OUTFILEFORMAT		FLOAT_DATA\n')
            c.write('LOGFILE ' + unwrap_name + '.log' + '\n')
            c.write('STATCOSTMODE SMOOTH')

        command = self.settings['snaphu_path'] + ' -f ' + conf_file
        os.system('cd ' + folder)
        os.system(command)

        unwrap_data = np.memmap(os.path.join(folder, out_file), np.float32, 'r', shape=shape)
        self['unwrapped'] = unwrap_data[:, :]
