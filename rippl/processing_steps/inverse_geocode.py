# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates


class InverseGeocode(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', out_coor=[], in_coor=[], dem_type='SRTM1', buffer=0, rounding=0, min_height=0,
                 max_height=0,
                 in_processes=[], in_file_types=[], in_data_ids=[], reference_slc='reference_slc', overwrite=False):

        """
        This function is used to find the line/pixel coordinates of the dem grid. These can later on be used to
        calculate the radar dem grid.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem out_coor: Coordinate system of the input grids. If dem_type is given this parameter can
                be left empty!
        :param CoordinateSystem in_coor: Coordinate system of the radar grid we should convert to
        :param str dem_type: In the case we want to use an imported DEM a dem_type should be defined. At the moment
                automatic generation of SRTM1 and SRTM3 are implemented, but also other DEMs can be imported manually.

        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData reference_slc: Image used to coregister the secondary_slc image for resampline etc.
        """

        """
        First define the name and output types of this processing step.
        1. process_name > name of process
        2. file_types > name of process types that will be given as output
        3. data_types > names 
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'inverse_geocode'
        self.output_info['image_type'] = 'reference_slc'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = ['lines', 'pixels']
        self.output_info['data_types'] = ['real4', 'real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['reference_slc']
        self.input_info['process_names'] = ['dem']
        self.input_info['file_names'] = ['dem']
        self.input_info['polarisations'] = ['']
        self.input_info['data_ids'] = [data_id]
        self.input_info['coor_types'] = ['out_coor']
        self.input_info['in_coor_types'] = ['']
        self.input_info['aliases_processing'] = ['dem']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor
        self.coordinate_systems['in_coor'] = in_coor
        
        # image data processing
        self.processing_images = dict()
        self.processing_images['reference_slc'] = reference_slc

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['dem_type'] = dem_type
        self.settings['in_coor'] = dict()
        self.settings['in_coor']['buffer'] = buffer
        self.settings['in_coor']['rounding'] = rounding
        self.settings['in_coor']['min_height'] = min_height
        self.settings['in_coor']['max_height'] = max_height

    def process_calculations(self):
        """
        We calculate the line and pixel coordinates of the DEM grid as a preparation for the conversion to a radar grid

        :return:
        """

        processing_data = self.processing_images['reference_slc']

        # Evaluate the orbit and create orbit coordinates object.
        in_coor = self.coordinate_systems['in_coor']
        out_coor = self.coordinate_systems['out_coor']
        orbit_interp = OrbitCoordinates(in_coor)

        if out_coor.grid_type == 'geographic':
            lats, lons = self.coordinate_systems['out_coor_chunk'].create_latlon_grid()
        elif out_coor.grid_type == 'projection':
            x, y = self.coordinate_systems['out_coor_chunk'].create_xy_grid()
            lats, lons = self.coordinate_systems['out_coor_chunk'].proj2ell(x, y)

        xyz = OrbitCoordinates.ell2xyz(np.ravel(lats), np.ravel(lons), np.ravel(self['dem']))

        lines, pixels = orbit_interp.xyz2lp(xyz)

        self['lines'] = np.reshape(lines, self.coordinate_systems['out_coor_chunk'].shape)
        self['pixels'] = np.reshape(pixels, self.coordinate_systems['out_coor_chunk'].shape)
        