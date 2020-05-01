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
import os

from drama.io import cfg
from drama.performance.sar import calc_aasr, calc_nesz, RASR, RASRdata, pattern, AASRdata, NESZdata, SARModeFromCfg
from scipy.interpolate import InterpolatedUnivariateSpline


class AzimuthAmbiguitiesLocations(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', out_coor=[], no_ambiguities=2, coreg_master='coreg_master', overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem in_coor: Coordinate system of the input grids.

        :param list[str] in_image_types: The type of the input ImageProcessingData objects (e.g. slave/master/ifg etc.)
        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'azimuth_ambiguities_locations'
        self.output_info['image_type'] = 'coreg_master'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = []
        self.output_info['data_types'] = []

        for dat_type in ['range', 'azimuth', 'gain']:
            for loc in ['left', 'right']:
                for amb_no in np.array(range(no_ambiguities)) + 1:
                    self.output_info['file_types'].append(dat_type + '_ambiguity_' + loc + '_no_' + str(amb_no))
                    self.output_info['data_types'].append('real4')

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['coreg_master']
        self.input_info['process_types'] = ['radar_ray_angles']
        self.input_info['file_types'] = ['incidence_angle']
        self.input_info['polarisations'] = ['']
        self.input_info['data_ids'] = [data_id]
        self.input_info['coor_types'] = ['out_coor']
        self.input_info['in_coor_types'] = ['']
        self.input_info['type_names'] = ['incidence_angles']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['coreg_master'] = coreg_master

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['no_ambiguities'] = no_ambiguities

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(AzimuthAmbiguitiesLocations, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        Calculate the lines and pixels using the orbit of the slave and coreg_master image.

        :return:
        """

        # Get the orbits
        orbit = self.processing_images['coreg_master'].find_best_orbit('original')
        readfile = self.processing_images['coreg_master'].readfiles['original']

        # Now initialize the orbit estimation.
        self.coordinate_systems['block_coor'].create_radar_lines()
        orbit_interp = OrbitCoordinates(coordinates=self.coordinate_systems['block_coor'], orbit=orbit)
        orbit_interp.lp_time()
        v_sat = np.sqrt(np.sum(orbit_interp.vel_orbit**2, axis=0))
        wavelength = readfile.wavelength
        PRF = readfile.json_dict['Pulse_repetition_frequency_raw_data (TOPSAR)']
        range_spacing = readfile.json_dict['Range_pixel_spacing']
        azimuth_spacing = readfile.json_dict['Azimuth_pixel_spacing']
        range_t = orbit_interp.ra_times

        aasr_values = self.get_max_gain_ambiguities(self.settings['no_ambiguities'])

        for amb_num in np.array(range(self.settings['no_ambiguities'])) + 1:
            azimuth_shift, range_shift = self.calculate_ambiguity_location(range_t, PRF, v_sat, wavelength, amb_num)

            for loc, change_loc in zip(['left', 'right'], [1, -1]):
                amb_str = '_ambiguity_' + loc + '_no_' + str(amb_num)

                self['azimuth' + amb_str] = change_loc * azimuth_shift / azimuth_spacing + orbit_interp.lines[:, None]
                self['azimuth' + amb_str][self['incidence_angles'] == 0] = 0
                self['range' + amb_str] = range_shift / range_spacing + orbit_interp.pixels[None, :]
                self['range' + amb_str][self['incidence_angles'] == 0] = 0
                self['gain' + amb_str][self['incidence_angles'] != 0] = aasr_values[loc][str(amb_num)](self['incidence_angles'][self['incidence_angles'] != 0])

    @staticmethod
    def calculate_ambiguity_location(range_t, PRF, v_sat, wavelength=0.05546576, amb_num=1):
        """
        Calculate the location of the ambiguities in range and azimuth

        We assume that the speed of the satellite is more or less the same everywhere. Otherwise the speed per azimuth
        cell should be specified.

        :return:
        """

        c = 299792458
        R0 = range_t / 2 * c

        az_distance = amb_num * (wavelength * PRF * R0[None, :]) / (2 * v_sat[:, None])
        ra_distance = np.sqrt(R0[None, :] ** 2 + az_distance ** 2) - R0

        return az_distance, ra_distance

    @staticmethod
    def get_max_gain_ambiguities(Namb):
        """
        Get the maximal gain from the ambiguities.

        :return:
        """

        # General setup
        main_dir = os.path.expanduser("~/surfdrive/TU_Delft/STEREOID/Data")
        rxname = 'airbus_dual_rx'
        txname = 'sentinel'
        is_bistatic = True
        runid = '2019_2'
        pardir = os.path.join(main_dir, 'PAR')
        pltdirr = os.path.join(os.path.join(os.path.join(main_dir, 'RESULTS'), 'Activation'), runid)
        parfile = os.path.join(pardir, ('Harmony_test.cfg'))
        conf = cfg.ConfigFile(parfile)
        mode = "IWS"
        Nswth = 3

        incidence_angles = np.linspace(15, 50, 351)

        # First calculate antenna patterns for aasr
        aasrs = []
        for swth in range(Nswth):
            aasr_ = calc_aasr(conf, mode, swth,
                              txname='sentinel',
                              rxname=rxname,
                              savedirr='',
                              t_in_bs=None,
                              n_az_pts=3,
                              view_patterns=False,
                              plot_patterns=False,
                              plot_AASR=False,
                              Tanalysis=20,
                              # vmin=-25.0, vmax=-15.0,
                              az_sampling=100, Namb=Namb,
                              bistatic=is_bistatic)
            aasrs.append(aasr_)

        aasr_values = {'left': dict(), 'right': dict()}

        # Sample for 0.1 degrees in inclination angle.
        for i_a, loc, amb_num in zip(range(Namb * 2), ['left', 'left', 'right', 'right'], [1, 2, 2, 1]):
            aasr_amb_vals = np.zeros(len(incidence_angles))
            for aasr in aasrs:

                aasr_vals = np.max(aasr.aasr_par[:, :, i_a], axis=0)
                aasr_amb_vals = np.maximum(np.interp(incidence_angles, aasr.inc_v, aasr_vals, left=0, right=0),
                                                 aasr_amb_vals)

            aasr_values[loc][str(amb_num)] = InterpolatedUnivariateSpline(incidence_angles, aasr_amb_vals, ext=3)

        return aasr_values
