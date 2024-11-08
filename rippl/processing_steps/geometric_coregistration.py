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

class GeometricCoregistration(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', in_coor=[], out_coor=[], dem_type='SRTM1',
                 in_image_types=[], in_processes=[], in_file_types=[], in_data_ids=[], only_DEM=False,
                 secondary_slc='secondary_slc', reference_slc='reference_slc', overwrite=False, coreg_crop=True):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem in_coor: Coordinate system of the input grids.

        :param list[str] in_image_types: The type of the input ImageProcessingData objects (e.g. secondary_slc/primary_slc/ifg etc.)
        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using a data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData secondary_slc: Secondary image, used as the default for input and output for processing.
        :param ImageProcessingData reference_slc: Image used to coregister the secondary_slc image for resampline etc.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'geometric_coregistration'
        self.output_info['image_type'] = 'secondary_slc'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = ['coreg_lines', 'coreg_pixels']
        self.output_info['data_types'] = ['real8', 'real8']

        # Input data information
        self.input_info = dict()
        if only_DEM:
            self.input_info['image_types'] = ['reference_slc']
            self.input_info['process_names'] = ['dem']
            self.input_info['file_names'] = ['dem']
            self.input_info['polarisations'] = ['']
            self.input_info['data_ids'] = [data_id]
            self.input_info['coor_types'] = ['out_coor']
            self.input_info['in_coor_types'] = ['']
            self.input_info['aliases_processing'] = ['dem']
        else:
            self.input_info['image_types'] = ['reference_slc', 'reference_slc', 'reference_slc']
            self.input_info['process_names'] = ['geocode', 'geocode', 'geocode']
            self.input_info['file_names'] = ['X', 'Y', 'Z']
            self.input_info['polarisations'] = ['', '', '',]
            self.input_info['data_ids'] = [data_id, data_id, data_id]
            self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor']
            self.input_info['in_coor_types'] = ['', '', '']
            self.input_info['aliases_processing'] = ['X_coreg', 'Y_coreg', 'Z_coreg']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor
        self.coordinate_systems['in_coor'] = in_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['secondary_slc'] = secondary_slc
        self.processing_images['reference_slc'] = reference_slc

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['coreg_crop'] = coreg_crop
        self.settings['only_DEM'] = only_DEM

    def process_calculations(self):
        """
        Calculate the lines and pixels using the orbit of the secondary_slc and reference_slc image.

        :return:
        """

        # Get the orbits
        orbit_secondary_slc = self.processing_images['secondary_slc'].find_best_orbit('original')
        orbit_reference_slc = self.processing_images['reference_slc'].find_best_orbit('original')

        # Now initialize the orbit estimation.
        if self.settings['coreg_crop']:
            orbit_interp = OrbitCoordinates(coordinates=self.coordinate_systems['in_coor_chunk'], orbit=orbit_secondary_slc)
        else:
            readfile_secondary_slc = self.processing_images['secondary_slc'].readfiles['original']
            coordinates = CoordinateSystem()
            coordinates.create_radar_coordinates()
            coordinates.load_readfile(readfile=readfile_secondary_slc)
            orbit_interp = OrbitCoordinates(coordinates=coordinates, orbit=orbit_secondary_slc)

        # Add DEM values
        if self.settings['only_DEM']:
            orbit_interp.height = self['dem']
            orbit_interp.lp_time()

            # Calculate cartesian coordinates
            orbit_interp.lph2xyz()
            xyz = orbit_interp.xyz
        else:
            xyz = np.vstack((np.ravel(self['X_coreg'])[None, :], np.ravel(self['Y_coreg'])[None, :], np.ravel(self['Z_coreg'])[None, :]))
        lines, pixels = orbit_interp.xyz2lp(xyz)

        self['coreg_lines'] = np.reshape(lines, self.coordinate_systems['out_coor_chunk'].shape)
        self['coreg_pixels'] = np.reshape(pixels, self.coordinate_systems['out_coor_chunk'].shape)
