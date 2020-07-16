# Try to do all calculations using numpy functions.
import numpy as np
import os

# Import the parent class Process for processing steps.
from rippl.meta_data.multilook_process import MultilookProcess
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates, OrbitInterpolate
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.user_settings import UserSettings
from rippl.external_dems.geoid import GeoidInterp
from rippl.resampling.grid_transforms import GridTransforms


class CalibratedAmplitudeMultilook(MultilookProcess):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='',
                 in_coor=[], out_coor=[], master_image=False, db_only=True, resampled=True,
                 coreg_master='coreg_master', slave='slave', overwrite=False, batch_size=1000000):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData master: Master image, generally used when creating interferograms
        :param ImageProcessingData ifg: Interferogram of a master/slave combination
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'calibrated_amplitude'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        if db_only:
            self.output_info['file_types'] = ['calibrated_amplitude_db']
            self.output_info['data_types'] = ['real4']
        else:
            self.output_info['file_types'] = ['calibrated_amplitude', 'calibrated_amplitude_db']
            self.output_info['data_types'] = ['real4', 'real4']
        self.db_only = db_only

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['slave']
        self.input_info['process_types'] = ['crop']
        self.input_info['file_types'] = ['crop']
        self.input_info['polarisations'] = [polarisation]
        self.input_info['data_ids'] = [data_id]
        self.input_info['coor_types'] = ['in_coor']
        self.input_info['in_coor_types'] = ['']
        self.input_info['type_names'] = ['complex_data']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = in_coor
        self.coordinate_systems['out_coor'] = out_coor

        # Image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['no_pre_calculated_coordinates'] = True
        self.settings['multilooked_grids'] = ['calibrated_amplitude']
        self.batch_size = batch_size

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(CalibratedAmplitudeMultilook, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def before_multilook_calculations(self):
        """
        This function creates an interferogram without additional multilooking.

        To find the lat/lon values we use the description given here:
        https://gamedev.stackexchange.com/questions/75756/sphere-sphere-intersection-and-circle-sphere-intersection

        :return:
        """

        orbit = self.processing_images['slave'].find_best_orbit('original')
        self.block_coor.create_radar_lines()        # type: CoordinateSystem
        az_time = self.block_coor.az_time
        az_step = self.block_coor.az_step
        ra_time = self.block_coor.ra_time
        ra_step = self.block_coor.ra_step
        center_lat = self.block_coor.center_lat
        center_lon = self.block_coor.center_lon

        R, geoid_h = self.globe_center_distance(center_lat, center_lon)

        # Line pixel coordinates
        lines = self.block_coor.interval_lines
        pixels = self.block_coor.interval_pixels
        az_times = az_time + az_step * lines
        c = 299792458
        ra_dist = (ra_time + ra_step * pixels) * c

        orbit_interp = OrbitInterpolate(orbit=orbit)
        xyz, vel_xyz, acc_xyz = orbit_interp.evaluate_orbit_spline(az_times, pos=False, vel=True, sorted=True)
        vel_n = vel_xyz / np.sqrt(np.sum(vel_xyz**2, axis=0))[None, :]

        # Calculate the globe intersection circles. Center location and sizes.
        circle_dist = np.einsum('ij,ij>j', vel_n, xyz)
        circle_center = circle_dist * vel_n
        circle_size = np.sqrt(R**2 - circle_dist**2)
        d = np.sqrt(np.sum((xyz - circle_center)**2 , axis=0))

        # Calculate tangent vector
        tangent_vector = np.einsum('ij,ij>ij',circle_center - xyz, vel_n)
        tangent_norm = tangent_vector / np.sqrt(np.sum(tangent_vector**2, axis=0))[None, :]

        # Now do the calcutations for all points.
        h = 0.5 + (ra_dist[None, :]**2 - circle_size[:, None]**2) / (2 * d[:, None])
        r_i = np.sqrt(ra_dist[None, :]**2 - h**2 * d[:, None]**2)
        c_i = xyz[:, :, None] + h[None, :, :] * (circle_center - xyz)[:, :, None]
        # Calc incidence angle using cosine rule
        incidence = np.pi - np.arccos((circle_size[:, None]**2 + ra_dist[None, :]**2 - d[:, None]**2) /
                              (2 * circle_size[None, :] * ra_dist[None, :]))
        del h

        # Now calculate locations in xyz coordinates
        coordinates = OrbitCoordinates()
        coordinates.height = geoid_h

        # Check for one point which side is the right one
        p0 = c_i[:, 0, 0] - tangent_norm[:, 0] * r_i[0, 0]
        coordinates.xyz = p0
        coordinates.xyz2ell()
        degree_dist0 = np.sqrt((center_lon - coordinates.lon)**2 +
                               (center_lat - coordinates.lat)**2)
        p1 = c_i[:, 0, 0] + tangent_norm[:, 0] * r_i[0, 0]
        coordinates.xyz = p1
        coordinates.xyz2ell()
        degree_dist1 = np.sqrt((center_lon - coordinates.lon)**2 +
                               (center_lat - coordinates.lat)**2)

        if degree_dist0 < degree_dist1:
            p_final = np.reshape(c_i - tangent_norm[:, :, None] * r_i[:, :], (3, r_i.size))
        elif degree_dist1 <= degree_dist0:
            p_final = np.reshape(c_i - tangent_norm[:, :, None] * r_i[:, :], (3, r_i.size))
        del r_i, c_i

        coordinates.xyz = p_final
        coordinates.xyz2ell()
        lat = coordinates.lat
        lon = coordinates.lon

        del coordinates, p_final

        # Final step is to calculate the lines and pixels in the final output.
        transform = GridTransforms(self.coordinate_systems['out_coor'], self.coordinate_systems['block_coor'])
        transform.add_dem(geoid_h * np.ones(lat.shape))
        transform.add_lat_lon(lat, lon)
        self['lines'], self['pixels'] = transform()

        beta_0 = 237.0  # TODO Read in from meta data files.
        self['calibrated_amplitude'] = np.abs(self['complex_data'])**2 * np.sin(incidence) / (beta_0 ** 2)

    def after_multilook_calculations(self):

        valid_pixels = self['calibrated_amplitude'] != 0

        # Calculate the db values.
        self['calibrated_amplitude_db'][valid_pixels] = 10 * np.log10(self['calibrated_amplitude'][valid_pixels])

    def globe_center_distance(self, lat, lon):
        """
        Calculate the distance from the center of the globe to the lat/lon location using the geoid file.

        """

        settings = UserSettings()
        settings.load_settings()

        geoid_file = os.path.join(settings.DEM_database, 'geoid', 'egm96.dat')
        geoid = GeoidInterp.create_geoid(egm_96_file=geoid_file, lat=np.array(lat), lon=np.array(lon))

        # Calculate location on WGS84 ellipsoid
        xyz = OrbitCoordinates.ell2xyz(lat, lon, geoid)
        R = np.sqrt(np.sum(xyz**2))

        return R, geoid