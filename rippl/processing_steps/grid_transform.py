# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem

from rippl.resampling.coor_new_extend import CoorNewExtend
from rippl.resampling.grid_transforms import GridTransforms


class GridTransform(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', in_coor=[], out_coor=[], in_coor_type='', out_coor_type='',
                 reference_slc='reference_slc', secondary_slc='secondary_slc', conversion_type='multilook',
                 use_xyz=True, use_latlon=True, use_dem=True, use_reference=True,
                 overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor

        :param ImageProcessingData reference_slc: Reference image, used as the default for input and output for processing.
        """

        if conversion_type not in ['multilook', 'resample']:
            raise ValueError('Choose either multilook or resample as convesion type')

        # Check in and out coor types
        for coor, coor_type in zip([in_coor, out_coor], [in_coor_type, out_coor_type]):
            if coor_type not in ['radar_coordinates', 'geographic', 'projected']:
                if isinstance(coor, CoordinateSystem):
                    coor_type = coor.grid_type

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'grid_transform'
        self.output_info['image_type'] = 'reference_slc'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        if conversion_type == 'multilook':
            self.output_info['file_names'] = ['multilook_lines', 'multilook_pixels']
        else:
            self.output_info['file_names'] = ['resample_lines', 'resample_pixels']
        self.output_info['data_types'] = ['real4', 'real4']

        xyz_input_info = dict()
        xyz_input_info['process_names'] = ['geocode', 'geocode', 'geocode']
        xyz_input_info['file_names'] = ['X', 'Y', 'Z']
        xyz_input_info['alias_processing'] = ['X', 'Y', 'Z']

        lat_lon_dem_input_info = dict()
        lat_lon_dem_input_info['process_names'] = ['dem', 'geocode', 'geocode']
        lat_lon_dem_input_info['file_names'] = ['dem', 'lat', 'lon']
        lat_lon_dem_input_info['alias_processing'] = ['dem', 'lat', 'lon']

        lat_lon_input_info = dict()
        lat_lon_input_info['process_names'] = ['geocode', 'geocode']
        lat_lon_input_info['file_names'] = ['lat', 'lon']
        lat_lon_input_info['alias_processing'] = ['lat', 'lon']

        dem_input_info = dict()
        dem_input_info['process_names'] = ['dem']
        dem_input_info['file_names'] = ['dem']
        dem_input_info['alias_processing'] = ['dem']

        no_input_info = dict()
        no_input_info['process_names'] = []
        no_input_info['file_names'] = []
        no_input_info['alias_processing'] = []

        # Coordinate systems
        self.coordinate_systems = dict()
        # For multilooking we want to know the positions of the old grid pixels in the new grid, while for the resampling
        # we want to know the new grid pixels in the old grid (irregular2regular vs regular2irregular)
        if conversion_type == 'multilook':
            self.coordinate_systems['in_coor'] = out_coor
            self.coordinate_systems['out_coor'] = in_coor
            in_type = out_coor_type
            out_type = in_coor_type
        elif conversion_type == 'resample':
            self.coordinate_systems['in_coor'] = in_coor
            self.coordinate_systems['out_coor'] = out_coor
            out_type = out_coor_type
            in_type = in_coor_type
        else:
            raise ValueError('conversion type should either be multilook or resample!')

        # Input data information
        self.input_info = dict()
        if in_type == 'radar_coordinates':
            if out_type == 'radar_coordinates':
                if use_xyz:
                    self.input_info = xyz_input_info
                elif use_latlon and use_dem:
                    self.input_info = lat_lon_dem_input_info
                elif use_latlon:        # In this case we assume we follow geoid height.
                    self.input_info = lat_lon_input_info
                elif use_dem:
                    self.input_info = dem_input_info
                else:
                    raise TypeError('At least the DEM or the lat/lon combination should be given to allow resampling.')
            else:
                if use_dem:
                    self.input_info = dem_input_info
                else:
                    self.input_info = no_input_info
        elif out_type == 'radar_coordinates':
            if use_latlon:
                self.input_info = lat_lon_input_info
            elif use_dem:
                self.input_info = dem_input_info
            else:
                raise TypeError('At least the DEM or the lat/lon combination should be given to allow resampling.')
        else:
            self.input_info = no_input_info

        # Generally we use the georeferenced information of the reference image. In some specific cases this is not
        # used. For example with amplitude images directly over oceans. Or if no interferograms are made.
        no_inputs = len(self.input_info['process_names'])
        if use_reference:
            self.input_info['image_types'] = ['reference_slc' for n in range(no_inputs)]
        else:
            self.input_info['image_types'] = ['secondary_slc' for n in range(no_inputs)]

        # For multilooking we want to use the output coordinates and for resampling the input coordinates
        if conversion_type == 'multilook':
            self.input_info['coor_types'] = ['out_coor' for n in range(no_inputs)]
        else:
            self.input_info['coor_types'] = ['in_coor' for n in range(no_inputs)]

        # Fill in other inputs
        self.input_info['polarisations'] = ['' for n in range(no_inputs)]
        self.input_info['data_ids'] = [data_id for n in range(no_inputs)]
        self.input_info['in_coor_types'] = ['' for n in range(no_inputs)]

        # If the shape of the input grid is not given we need information from an input file.
        self.overwrite = overwrite

        # image data processing
        self.processing_images = dict()
        self.processing_images['reference_slc'] = reference_slc
        self.processing_images['secondary_slc'] = secondary_slc

        self.settings = dict()
        self.settings['conversion_type'] = conversion_type

    def process_calculations(self):
        """
        This step contains all the calculations that are done for this step. This is therefore the step that is most
        specific for every process definition.
        To run the full processing you have to initialize (__init__) and call (__call__) the object.

        To access the input data and save the output data you can the images' dictionary. This contains all the
        memory data files. The keys are the same as the file_types and in_file_types.

        :return:
        """

        transform = GridTransforms(self.coordinate_systems['in_coor_chunk'], self.coordinate_systems['out_coor_chunk'])

        if self.input_info['file_names'] == ['X', 'Y', 'Z']:
            transform.add_xyz(self['X'], self['Y'], self['Z'])
            valid = (self['X'] != 0) * (self['Y'] != 0) * (self['Z'] != 0)
        elif self.input_info['file_names'] == ['dem', 'lat', 'lon']:
            transform.add_dem(self['dem'])
            transform.add_lat_lon(self['lat'], self['lon'])
            valid = (self['lat'] != 0) * (self['lon'] != 0) * (self['dem'] != 0)
        elif self.input_info['file_names'] == ['lat', 'lon']:
            transform.add_lat_lon(self['lat'], self['lon'])
            valid = (self['lat'] != 0) * (self['lon'] != 0)
        elif self.input_info['file_names'] == ['dem']:
            transform.add_dem(self['dem'])
            valid = (self['dem'] != 0)
        else:
            valid = []

        lines, pixels = transform()
        if len(valid) == 0:
            valid = np.ones(lines.shape, dtype=bool)

        if self.settings['conversion_type'] == 'multilook':
            self['multilook_lines'][valid] = lines[valid]
            self['multilook_pixels'][valid] = pixels[valid]
        else:
            self['resample_lines'][valid] = lines[valid]
            self['resample_pixels'][valid] = pixels[valid]

    def def_out_coor(self):
        """
        Define output coordinate grid.

        :return:
        """

        self.coordinate_systems['in_coor'] = CoorNewExtend(self.coordinate_systems['out_coor'],
                                                           self.coordinate_systems['in_coor']).out_coor
