"""
Function to calculate the latitude/longitude of the pixels at a certain height. This is usefull to calculate for example
tropospheric and ionospheric delays.

"""

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
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates, OrbitInterpolate
from rippl.meta_data.readfile import Readfile


class GeocodeHeight(Process):  # Change this name to the one of your processing step.

    def __init__(self, out_coor=[], coreg_master='coreg_master', overwrite=False, height=350):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem in_coor: Coordinate system of the input grids.

        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'geocode_height'
        self.output_info['image_type'] = 'coreg_master'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = str(height) + '_km'
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['lat', 'lon']
        self.output_info['data_types'] = ['real4', 'real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['coreg_master', 'coreg_master', 'coreg_master', 'coreg_master']
        self.input_info['process_types'] = ['geocode', 'geocode', 'geocode', 'radar_ray_angles']
        self.input_info['file_types'] = ['X', 'Y', 'Z', 'incidence_angle']
        self.input_info['polarisations'] = ['', '', '', '']
        self.input_info['data_ids'] = ['', '', '', '']
        self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '', '']
        self.input_info['type_names'] = ['X', 'Y', 'Z', 'incidence_angle']

        self.overwrite = overwrite

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['coreg_master'] = coreg_master
        self.settings = dict()
        self.settings['height'] = dict()

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(GeocodeHeight, self).__init__(
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
        readfile = self.processing_images['coreg_master'].readfiles['original']
        self.block_coor.create_radar_lines()
        orbit_interp = OrbitInterpolate(orbit)

        # Calculate the sensing azimuth times.
        az_first_pix_time, date = Readfile.time2seconds(readfile.json_dict['First_pixel_azimuth_time (UTC)'])
        az_time_step = readfile.json_dict['Pulse_repetition_frequency_raw_data (TOPSAR)']

        az_times = self.block_coor.ml_lines * az_time_step + az_first_pix_time
        xyz_orbit = orbit_interp.evaluate_orbit_spline(az_times, True, False, False, True)
        num_points = self.block_coor.shape[0] * self.block_coor.shape[1]

        # Calc xyz and velocity vector of satellite orbit.
        xyz_diff = np.vstack(np.reshape(xyz_orbit[0, :, None] - self['X'][None, :, :], (1, num_points)),
                             np.reshape(xyz_orbit[1, :, None] - self['Y'][None, :, :], (1, num_points)),
                             np.reshape(xyz_orbit[2, :, None] - self['Z'][None, :, :], (1, num_points)))
        xyz_new = xyz_diff / np.sqrt(np.sum(xyz_diff**2), axis=0) * self.settings['height'] + \
                  np.vstack(np.ravel(self['X'])[None, :],
                            np.ravel(self['Y'])[None, :],
                            np.ravel(self['Z'])[None, :])

        orbit_interp = OrbitCoordinates(coordinates=self.block_coor, readfile=readfile)
        orbit_interp.xyz = xyz_new
        orbit_interp.xyz2ell()
        self['lat'] = np.reshape(orbit_interp.lat, self.block_coor.shape)
        self['lon'] = np.reshape(orbit_interp.lon, self.block_coor.shape)
