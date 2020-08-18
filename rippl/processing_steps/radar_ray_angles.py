# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates


class RadarRayAngles(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', out_coor=[], coreg_master='coreg_master', overwrite=False, squint=False,
                 off_nadir_angle=True, heading=True, incidence_angle=True, azimuth_angle=True):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem in_coor: Coordinate system of the input grids.

        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'radar_ray_angles'
        self.output_info['image_type'] = 'coreg_master'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = []
        self.output_info['data_types'] = []
        if off_nadir_angle:
            self.output_info['file_types'].append('off_nadir_angle')
            self.output_info['data_types'].append('real4')
        if heading:
            self.output_info['file_types'].append('heading')
            self.output_info['data_types'].append('real4')
        if incidence_angle:
            self.output_info['file_types'].append('incidence_angle')
            self.output_info['data_types'].append('real4')
        if azimuth_angle:
            self.output_info['file_types'].append('azimuth_angle')
            self.output_info['data_types'].append('real4')
        if squint:
            self.output_info['file_types'].append('squint_angle')
            self.output_info['data_types'].append('real2')

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['coreg_master', 'coreg_master', 'coreg_master', 'coreg_master']
        self.input_info['process_types'] = ['geocode', 'geocode', 'geocode', 'dem']
        self.input_info['file_types'] = ['X', 'Y', 'Z', 'dem']
        self.input_info['data_types'] = ['real4', 'real4', 'real4', 'real4']
        self.input_info['polarisations'] = ['', '', '', '']
        self.input_info['data_ids'] = [data_id, data_id, data_id, data_id]
        self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '', '']
        self.input_info['type_names'] = ['X', 'Y', 'Z', 'dem']

        self.overwrite = overwrite

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['coreg_master'] = coreg_master
        self.settings = dict()
        self.settings['squint'] = squint
        self.settings['off_nadir_angle'] = off_nadir_angle
        self.settings['incidence_angle'] = incidence_angle
        self.settings['heading'] = heading
        self.settings['azimuth_angle'] = azimuth_angle

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(RadarRayAngles, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        Here we calculate the angles that the ray between the satellite and a point on earth make with respect to the
        earths surface.
        The calculated angles are:
        - incidence angle - the angle of the radar signal at the satellite wrt the line to the earth centre.
        - heading - the azimuth angle at which the satellite travels
        - elevation angle - the angle of the radar signal at the point on the ground w.r.t. the earth ellipsoid
        - azimuth angle - the azimuth angle of a point on the ground towards the satellite

        :return:
        """

        # Get the orbit and initialize orbit coordinates
        orbit = self.processing_images['coreg_master'].find_best_orbit('original')
        self.block_coor.create_radar_lines()
        orbit_interp = OrbitCoordinates(coordinates=self.block_coor, orbit=orbit)

        # Calc xyz and velocity vector of satellite orbit.
        orbit_interp.height = self['dem']
        orbit_interp.xyz = np.vstack((np.ravel(self['X'])[None, :], np.ravel(self['Y'])[None, :], np.ravel(self['Z'])[None, :]))

        if self.block_coor.grid_type != 'radar_coordinates':
            lines, pixels = orbit_interp.xyz2lp(orbit_interp.xyz)
            orbit_interp.lp_time(lines, pixels, regular=False)
        else:
            orbit_interp.lp_time()

        # Calc angles based on xyz information from geocoding
        if self.settings['incidence_angle'] or self.settings['azimuth_angle']:
            orbit_interp.xyz2scatterer_azimuth_elevation()
        if self.settings['off_nadir_angle'] or self.settings['heading']:
            orbit_interp.xyz2orbit_heading_off_nadir()
        if self.settings['squint']:
            orbit_interp.xyz2squint()

        if self.settings['off_nadir_angle']:
            self['off_nadir_angle'] = np.reshape(orbit_interp.off_nadir_angle, self.block_coor.shape)
        if self.settings['heading']:
            self['heading'] = np.reshape(orbit_interp.heading, self.block_coor.shape)
        if self.settings['incidence_angle']:
            self['incidence_angle'] = np.reshape(90 - orbit_interp.elevation_angle, self.block_coor.shape)
        if self.settings['azimuth_angle']:
            self['azimuth_angle'] = np.reshape(orbit_interp.azimuth_angle, self.block_coor.shape)
