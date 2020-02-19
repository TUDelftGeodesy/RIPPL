# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

"""
Test

self = NeszHarmonySentinel(nesz_main_dir='/home/gert/Surfdrive/TU_Delft/STEREOID/Data')
self.swth = 0

"""

# Try to do all calculations using numpy functions.
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.orbit_geometry.orbit_interpolate import OrbitInterpolate
from rippl.processing_steps.deramp import Deramp

import drama.geo.sar as geo
from drama.performance.sar import calc_aasr, calc_nesz, RASR, pattern, AASRdata, RASRdata, NESZdata, SARModeFromCfg
from drama.performance.sar.azimuth_performance import mode_from_conf
from drama.geo.sar.geo_history import GeoHistory
from drama.io import cfg

import stereoid.sar_performance as strsarperf
import stereoid.oceans as strocs


class NeszHarmonySentinel(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', out_coor=[],
                 in_processes=[], in_file_types=[], in_data_ids=[],
                 coreg_master='coreg_master', overwrite=False, nesz_main_dir=''):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'nesz'
        self.output_info['image_type'] = 'coreg_master'
        self.output_info['polarisation'] = ''
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['nesz_harmony_ati', 'nesz_harmony_dual', 'nesz_sentinel']
        self.output_info['data_types'] = ['real4', 'real4', 'real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['coreg_master']
        self.input_info['process_types'] = ['radar_ray_angles']
        self.input_info['file_types'] = ['incidence_angle']
        self.input_info['polarisations'] = ['']
        self.input_info['data_ids'] = [data_id]
        self.input_info['coor_types'] = ['out_coor']
        self.input_info['in_coor_types'] = ['']
        self.input_info['type_names'] = ['incidence_angle']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['coreg_master'] = coreg_master

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.main_dir = nesz_main_dir

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(NeszHarmonySentinel, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        With this function we calculate the NESZ for the original Sentinel data and Harmony mission in ATI and dual mode.

        :return:
        """
        processing_data = self.processing_images['coreg_master']
        coordinates = self.coordinate_systems['in_block_coor']
        readfile = processing_data.readfiles['original']   

        # Calculate the time difference of every pixel w.r.t. the doppler centroid
        az_times = Deramp.az_ra_time(self.coordinate_systems['block_coor'])[0] - readfile.DC_ref_az[0]
        self.swth = readfile.swath - 1
        
        # Interpolate the NESZ values based on input values and other data using bilinear interpolation.
        self.general_settings(rxname='airbus_dual_rx')
        nesz = self.calc_nesz(do_nesz=False)
        nesz_interpolator = RectBivariateSpline(nesz.t_in_burst, nesz.inc_v, nesz.nesz)
        self['nesz_harmony_dual'] = nesz_interpolator.ev(az_times, self['incidence_angle'])

        self.general_settings(rxname='airbus_ati_rx')
        nesz = self.calc_nesz(do_nesz=False)
        nesz_interpolator = RectBivariateSpline(nesz.t_in_burst, nesz.inc_v, nesz.nesz)
        self['nesz_harmony_ati'] = nesz_interpolator.ev(az_times, self['incidence_angle'])

        self.general_settings(rxname='sentinel')
        nesz = self.calc_nesz(do_nesz=False)
        nesz_interpolator = RectBivariateSpline(nesz.t_in_burst, nesz.inc_v, nesz.nesz)
        self['nesz_sentinel'] = nesz_interpolator.ev(az_times, self['incidence_angle'])

    def general_settings(self, rxname='tud_triple_rx'):
        """
        Load some general settings to calculate NESZ/AASR/RASR

        :return:
        """

        self.rxname = rxname
        self.txname = 'sentinel'
        # For companions, is_bistatic should be True
        if rxname == 'sentinel':
            self.is_bistatic = False
        else:
            self.is_bistatic = True
        runid = 'EUSAR'
        pardir = os.path.join(self.main_dir, 'PAR')
        pltdirr = os.path.join(os.path.join(os.path.join(self.main_dir, 'RESULTS'), 'Activation'), runid)
        parfile = os.path.join(pardir, ("STEREOID_%s.cfg" % runid))
        self.conf = cfg.ConfigFile(parfile)
        # extract relevant info from conf
        self.rxcnf = getattr(self.conf, self.rxname)
        txcnf = getattr(self.conf, self.txname)
        if self.is_bistatic:
            dau_km = int(self.conf.formation_primary.dau[0] / 1e3)
        else:
            dau_km = int(0)
        dau_str = ("%ikm" % dau_km)
        indstr = self.rxname
        sysid = indstr  # ("%s_%3.2fm" % (indstr, b_at))
        if self.rxcnf.DBF:
            sysid = sysid + "_DBF"

        if self.rxcnf.SCORE:
            sysid = sysid + "_SCORE"

        self.savedirr = os.path.join(os.path.join(os.path.join(self.main_dir, 'RESULTS'), 'SARPERF'), sysid)
        self.savedirr = os.path.join(self.savedirr, dau_str)
        self.mode = "IWS"  #
        # mode = "stripmap"
        self.n_az_pts = 100
        inc_range = [10, 45]
        (incs, PRFs, proc_bw,
         steering_rates,
         burst_lengths, self.short_name, proc_tap, tau_p, bw_p) = mode_from_conf(self.conf, self.mode)


    def calc_aasr(self, do_aasr=True):
        """
        Calculate AASR pattern for burst

        :return:
        """

        modeandswth = ("%s_sw%i" % (self.short_name, self.swth + 1))
        modedir = os.path.join(self.savedirr, modeandswth)
        if not os.path.exists(os.path.join(modedir, 'aasr.p')) or do_aasr:
            aasr = calc_aasr(self.conf, self.mode, self.swth,
                              txname='sentinel',
                              rxname=self.rxname,
                              savedirr=self.savedirr,
                              t_in_bs=None,
                              n_az_pts=self.n_az_pts,
                              view_patterns=False,
                              vmin=-20.0, vmax=-15.0,
                              az_sampling=500, Namb=3,
                              bistatic=self.is_bistatic)
            aasr.save(os.path.join(modedir, 'aasr.p'))
        else:
            aasr = AASRdata.from_file(os.path.join(modedir, 'aasr.p'))

        return aasr

    def calc_rasr(self, do_rasr=False):
        """
        Calculate RASR pattern for burst

        :return:
        """

        modeandswth = ("%s_sw%i" % (self.short_name, self.swth + 1))
        modedir = os.path.join(self.savedirr, modeandswth)
        if not os.path.exists(os.path.join(modedir,'rasr.p')) or do_rasr:
            rasr = RASR(self.conf, self.mode, self.swth, txname='sentinel',
                         rxname=self.rxname,
                         savedirr=self.savedirr,
                         t_in_bs=None,
                         n_az_pts=self.n_az_pts, n_amb_az=0, Tanalysis=10,
                         vmin=-25, vmax=-18, az_sampling=500, bistatic=self.is_bistatic, verbosity=1)

            plt.close('all')
            rasr.save(os.path.join(modedir, 'rasr.p'))
        else:
            rasr = RASRdata.from_file(os.path.join(modedir, 'rasr.p'))

        return rasr

    def calc_nesz(self, do_nesz=False):
        """
        Calculate NESZ pattern for burst

        :return:
        """

        modeandswth = ("%s_sw%i" % (self.short_name, self.swth + 1))
        modedir = os.path.join(self.savedirr, modeandswth)
        if not os.path.exists(os.path.join(modedir,'nesz.p')) or do_nesz:
            if self.rxname == 'sentinel':
                nesz = calc_nesz(self.conf, self.mode, self.swth, txname='sentinel',
                                  rxname=self.rxname,
                                  savedirr=self.savedirr,
                                  t_in_bs=None,
                                  n_az_pts=self.n_az_pts,
                                  vmin=-29, vmax=-17, az_sampling=500, bistatic=self.is_bistatic)
            else:
                nesz = calc_nesz(self.conf, self.mode, self.swth, txname='sentinel',
                                  rxname=self.rxname,
                                  savedirr=self.savedirr,
                                  t_in_bs=None,
                                  n_az_pts=self.n_az_pts,
                                  extra_losses=self.rxcnf.L,
                                  vmin=-25, vmax=-17, az_sampling=500, bistatic=self.is_bistatic)
            plt.close('all')
            nesz.save(os.path.join(modedir,'nesz.p'))
        else:
            nesz = NESZdata.from_file(os.path.join(modedir,'nesz.p'))

        return nesz
