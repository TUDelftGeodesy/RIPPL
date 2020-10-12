# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.meta_data.orbit import Orbit
from rippl.orbit_geometry.orbit_interpolate import OrbitInterpolate


class Deramp(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', out_coor=[], slave='slave', overwrite=False):
        """
        This function deramps the ramped data from TOPS mode to a deramped data. Input data of this function should
        be a radar coordinates grid.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs
        :param CoordinateSystem out_coor: Coordinate system of the input grids.
        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'deramp'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['deramped']
        self.output_info['data_types'] = ['complex_real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['slave']
        self.input_info['process_types'] = ['crop']
        self.input_info['file_types'] = ['crop']
        self.input_info['polarisations'] = [polarisation]
        self.input_info['data_ids'] = [data_id]
        self.input_info['coor_types'] = ['out_coor']
        self.input_info['in_coor_types'] = ['']
        self.input_info['type_names'] = ['ramped_data']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(Deramp, self).__init__(
                       input_info=self.input_info,
                       output_info=self.output_info,
                       coordinate_systems=self.coordinate_systems,
                       processing_images=self.processing_images,
                       overwrite=self.overwrite,
                       settings=self.settings)

    def process_calculations(self, test=False):
        """
        Main calculations done are based on information given in the meta data. The ramp can be calculated based on the
        DC and FM polynomials combined with range and azimuth time.

        Reference to the algorithm can be found here:
        https://earth.esa.int/documents/247904/1653442/Sentinel-1-TOPS-SLC_Deramping

        :return:
        """
        
        processing_data = self.processing_images['slave']
        if not isinstance(processing_data, ImageProcessingData):
            print('Input data missing')

        coordinates = self.coordinate_systems['in_block_coor']
        readfile = processing_data.readfiles['original']      
        orbit = processing_data.find_best_orbit('original')

        # Calculate azimuth/range grid and ramp.
        az_grid, ra_grid = self.az_ra_time(coordinates)
        ramp = self.calc_ramp(readfile, orbit, az_grid, ra_grid)

        # Finally calced the deramped image.
        self['deramped'] = self['ramped_data'] * ramp

        if test:
            self.test_result()

    def test_result(self):
        """
        Method to check the results of the deramping. Here we generate the spectrogram of the input and output and
        create two images.

        :return:
        """

        # import needed function
        import matplotlib.pyplot as plt

        # Calculate the spectrogram of the ramped data
        spec_in = np.log(np.abs(np.fft.ifftshift(np.fft.fft2(self['ramped_data']))**2))

        # And the deramped data
        spec_out = np.log(np.abs(np.fft.ifftshift(np.fft.fft2(self['deramped'])) ** 2))

        # plot in one image
        plt.figure()
        plt.subplot(211)
        plt.imshow(spec_in[:, ::10])

        plt.subplot(212)
        plt.imshow(spec_out[:, ::10])
        plt.show()

    # Next are two helper functions to calculate the ramp for the deramping. Note that deramping only works for
    # a radar grid.
    @staticmethod
    def az_ra_time(coordinates):
        """
        Find the azimuth and range time for all pixels.

        :param CoordinateSystem coordinates: coordinates of radar grid to calculate a grid of azimuth and range times.
        :return: grids of azimuth and range times
        :rtype: np.ndarray
        """

        if coordinates.grid_type != 'radar_coordinates':
            print('Deramping for non radar grid not possible.')
            return

        # Get the lines/pixels from the coordinate system.
        coordinates.create_radar_lines()
        az_times = coordinates.ml_lines * coordinates.az_step + coordinates.az_time
        ra_times = coordinates.ml_pixels * coordinates.ra_step + coordinates.ra_time

        # Now create the range and azimuth time grids.
        ra_time_grid, az_time_grid = np.meshgrid(ra_times, az_times)

        return az_time_grid, ra_time_grid

    @staticmethod
    def calc_ramp(readfile, orbit, az_time_grid, ra_time_grid, demodulation=False):
        """
        Calculate the phase ramp for deramping/reramping
        
        :param Readfile readfile: Readfile object for this dataset
        :param Orbit orbit: Orbit object for the same readfile
        :param np.ndarray az_time_grid: Grid with azimuth times
        :param np.ndarray ra_time_grid: Grid with range times
        :param bool demodulation: Do you want to perform demodulation (default is False)
        :return: ramp due steering of radar antenna
        :rtype: np.ndarray
        """

        interp_orbit = OrbitInterpolate(orbit)
        interp_orbit.fit_orbit_spline()
        mid_orbit_time = readfile.orig_az_first_pix_time + readfile.az_time_step * (float(readfile.size[0]) / 2)
        mid_orbit_range_time = readfile.orig_ra_first_pix_time + readfile.ra_time_step * (float(readfile.size[1]) / 2)
        az_time_grid -= mid_orbit_time
        orbit_vel = interp_orbit.evaluate_orbit_spline(np.asarray([mid_orbit_time]), pos=False, vel=True, acc=False)[1]

        orbit_velocity = np.sqrt(orbit_vel[0] ** 2 + orbit_vel[1] ** 2 + orbit_vel[2] ** 2)

        # Compute Nominal DC for the whole burst
        # Compute FM rate along range
        k_fm = readfile.FM_polynomial[0] + readfile.FM_polynomial[1] * (ra_time_grid - readfile.FM_ref_ra) + readfile.FM_polynomial[2] * (ra_time_grid - readfile.FM_ref_ra) ** 2
        k_fm_0 = (readfile.FM_polynomial[0] + readfile.FM_polynomial[1] * (mid_orbit_range_time - readfile.FM_ref_ra) + readfile.FM_polynomial[2] *
                  (mid_orbit_range_time - readfile.FM_ref_ra) ** 2)

        # Compute DC along range at reference azimuth time (azimuthTime)
        df_az_ctr = readfile.DC_polynomial[0] + readfile.DC_polynomial[1] * (ra_time_grid - readfile.DC_ref_ra) + readfile.DC_polynomial[2] * (ra_time_grid - readfile.DC_ref_ra) ** 2
        f_dc_ref_0 = (readfile.DC_polynomial[0] + readfile.DC_polynomial[1] * (mid_orbit_range_time - readfile.DC_ref_ra) + readfile.DC_polynomial[2] *
                      (mid_orbit_range_time - readfile.DC_ref_ra) ** 2)
        ra_time_grid = []

        # From S-1 steering rate and orbit information
        # Computes sensor velocity from orbits

        # Frequency rate
        ks_hz = 2 * np.mean(orbit_velocity) / readfile.wavelength * readfile.steering_rate / 180 * np.pi

        # Time ratio
        alpha_nom = 1 - ks_hz / k_fm

        # DC Azimuth rate [Hz/s]
        dr_est = ks_hz / alpha_nom
        ks_hz = []
        alpha_nom = []

        # Reference time
        az_dc = -(df_az_ctr / k_fm) + (f_dc_ref_0 / k_fm_0)
        k_fm = []
        k_fm_0 = []
        t_az_vec = az_time_grid - az_dc
        az_dc = []

        # Generate inverse chirp
        if demodulation:
            ramp = np.exp(-1j * np.pi * dr_est * t_az_vec ** 2 + -1j * np.pi * 2 * df_az_ctr * t_az_vec).astype(np.complex64)
        else:
            ramp = np.exp(-1j * np.pi * dr_est * t_az_vec ** 2).astype(np.complex64)

        return ramp
