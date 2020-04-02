# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem

from rippl.resampling.coor_new_extend import CoorNewExtend
from rippl.resampling.grid_transforms import GridTransforms


class ResamplePrepare(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', in_coor=[], out_coor=[],
                 in_image_type='slave', in_process='', in_file_type='', in_polarisation='', in_data_id='',
                 out_image_type='slave', out_process='', out_file_type='', out_polarisation='', out_data_id='',
                 coreg_master='coreg_master', slave='slave', conversion_type='multilook', overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor

        :param ImageProcessingData coreg_master: Coreg master image, used as the default for input and output for processing.
        """

        if conversion_type not in ['multilook', 'resample']:
            raise ValueError('Choose either multilook or resample as convesion type')

        self.conversion_type = conversion_type

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'reproject'
        self.output_info['image_type'] = 'coreg_master'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['out_coor_lines', 'out_coor_pixels']
        self.output_info['data_types'] = ['real4', 'real4']

        # Input data information
        self.input_info = dict()
        if in_coor.grid_type == 'radar_coordinates':
            if out_coor.grid_type == 'radar_coordinates':
                self.input_info['image_types'] = ['coreg_master', 'coreg_master', 'coreg_master', in_image_type]
                self.input_info['process_types'] = ['geocode', 'geocode', 'geocode', in_process]
                self.input_info['file_types'] = ['X', 'Y', 'Z', in_file_type]
                self.input_info['polarisations'] = ['', '', '', in_polarisation]
                self.input_info['data_ids'] = [data_id, data_id, data_id, in_data_id]
                self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor', 'in_coor']
                self.input_info['in_coor_types'] = ['', '', '', '', '']
                self.input_info['type_names'] = ['X', 'Y', 'Z', 'in_coor_grid']
            else:
                self.input_info['image_types'] = ['coreg_master', in_image_type]
                self.input_info['process_types'] = ['dem', in_process]
                self.input_info['file_types'] = ['dem', in_file_type]
                self.input_info['polarisations'] = ['', '', '', in_polarisation]
                self.input_info['data_ids'] = [data_id, in_data_id]
                self.input_info['coor_types'] = ['out_coor', 'in_coor']
                self.input_info['in_coor_types'] = ['', '']
                self.input_info['type_names'] = ['dem', 'in_coor_grid']
        else:
            if out_coor.grid_type == 'radar_coordinates':
                self.input_info['image_types'] = ['coreg_master', 'coreg_master', 'coreg_master', in_image_type]
                self.input_info['process_types'] = ['dem', 'geocode', 'geocode', in_process]
                self.input_info['file_types'] = ['dem', 'lat', 'lon', in_file_type]
                self.input_info['polarisations'] = ['', '', '', in_polarisation]
                self.input_info['data_ids'] = [data_id, data_id, data_id, in_data_id]
                self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor', 'in_coor']
                self.input_info['in_coor_types'] = ['', '', '', '']
                self.input_info['type_names'] = ['dem', 'lat', 'lon', 'in_coor_grid']
            else:
                self.input_info['image_types'] = [in_image_type]
                self.input_info['process_types'] = [in_process]
                self.input_info['file_types'] = [in_file_type]
                self.input_info['polarisations'] = [in_polarisation]
                self.input_info['data_ids'] = [in_data_id]
                self.input_info['coor_types'] = ['in_coor']
                self.input_info['in_coor_types'] = ['']
                self.input_info['type_names'] = ['in_coor_grid']
                if out_process:
                    self.input_info['image_types'].append(out_image_type)
                    self.input_info['process_types'].append(out_process)
                    self.input_info['file_types'].append(out_file_type)
                    self.input_info['polarisations'].append(out_polarisation)
                    self.input_info['data_ids'].append(out_data_id)
                    self.input_info['coor_types'].append('out_coor')
                    self.input_info['in_coor_types'].append('')
                    self.input_info['type_names'].append('out_coor_grid')

        # If the shape of the input grid is not given we need information from an input file.
        self.overwrite = overwrite

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = in_coor
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['coreg_master'] = coreg_master
        self.processing_images['slave'] = slave
        self.settings = dict()

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(ResamplePrepare, self).__init__(
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

        transform = GridTransforms(self.coordinate_systems['in_block_coor'], self.coordinate_systems['block_coor'])

        if self.coordinate_systems['out_coor'].grid_type == 'radar_coordinates':
            if self.coordinate_systems['in_coor'].grid_type == 'radar_coordinates':
                transform.add_xyz(self['X'], self['Y'], self['Z'])
            elif self.coordinate_systems['in_coor'].grid_type in ['projection', 'geographic']:
                transform.add_dem(self['dem'])
                transform.add_lat_lon(self['lat'], self['lon'])

        self['out_coor_lines'], self['out_coor_pixels'] = transform()

    def def_out_coor(self):
        """
        Define output coordinate grid.

        :return:
        """

        if self.coordinate_systems['out_coor'].shape == [0, 0]:
            new_coor = CoorNewExtend(self.coordinate_systems['in_coor'], self.coordinate_systems['out_coor'])
            self.coordinate_systems['out_coor'] = new_coor.out_coor
