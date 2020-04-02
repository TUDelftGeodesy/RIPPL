
from rippl.meta_data.stack import Stack
from rippl.meta_data.image_processing_data import ImageProcessingData, ImageProcessingMeta
from rippl.SAR_sensors.sentinel.sentinel_stack import SentinelStack
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.SAR_sensors.sentinel.sentinel_download import DownloadSentinelOrbit, DownloadSentinel
from rippl.resampling.coor_new_extend import CoorNewExtend
import os
import numpy as np
import copy

from rippl.processing_steps.resample import Resample
from rippl.processing_steps.import_dem import ImportDem
from rippl.processing_steps.inverse_geocode import InverseGeocode
from rippl.processing_steps.resample_dem import ResampleDem
from rippl.processing_steps.geocode import Geocode
from rippl.processing_steps.geometric_coregistration import GeometricCoregistration
from rippl.processing_steps.earth_topo_phase import EarthTopoPhase
from rippl.processing_steps.square_amplitude import SquareAmplitude
from rippl.processing_steps.reramp import Reramp
from rippl.processing_steps.radar_ray_angles import RadarRayAngles
from rippl.processing_steps.baseline import Baseline
from rippl.processing_steps.coherence import Coherence
from rippl.processing_steps.deramp_resample_radar_grid import DerampResampleRadarGrid
from rippl.processing_steps.interferogram_multilook import InterferogramMultilook
from rippl.processing_steps.square_amplitude_multilook import SquareAmplitudeMultilook
from rippl.processing_steps.calibrated_amplitude_multilook import CalibratedAmplitudeMultilook
from rippl.processing_steps.multilook_prepare import MultilookPrepare
from rippl.processing_steps.resample_prepare import ResamplePrepare
from rippl.processing_steps.unwrap import Unwrap
from rippl.processing_steps.nesz_harmony_sentinel import NeszHarmonySentinel

from rippl.processing_steps.multilook import Multilook
from rippl.pipeline import Pipeline

class GeneralPipelines():


    def __init__(self, processes=6):
        """
        Initialize the general S1 processing chain.

        """

        self.radar_coor = CoordinateSystem()
        self.radar_coor.create_radar_coordinates()

        self.processes = processes

    def get_data(self, data_type, slice=False, concat_meta=False):
        """
        This function loads the data for the different data types (ifg, coreg_master, coreg_slave, slc)

        In case the concat_meta is False
        If slice is True all slices are loaded, otherwise only the full images.
        - slc > All slcs in the stack are loaded. If you want to work on coregistration or similar it is better to use
                the coreg_slave, as it gives the coreg_master images alongside these images.
        - ifg > All the ifg images of the stack. It will also add master, slave and coreg_master
        - coreg_master > The coreg master of the stack, or the slices that are part of it
        - coreg_slave > Get all the slave images and the corresponding coreg_master images.

        :param data_type:
        :param slice:
        :return:
        """

        coreg_master_name = self.stack.master_date
        slice_names = np.sort(self.stack.slcs[coreg_master_name].slice_names)

        if data_type == 'slc':
            slc_names = list(self.stack.slcs.keys())
            images = []
            if slice:
                for slc_name in slc_names:
                    for slice_name in slice_names:
                        if slice_name in self.stack.slcs[slc_name].slice_data.keys():
                            images.append(self.stack.slcs[slc_name].slice_data[slice_name])
            else:
                for slc_name in slc_names:
                    if concat_meta:
                        images.append(self.stack.slcs[slc_name])
                    else:
                        images.append(self.stack.slcs[slc_name].data)

            return images

        elif data_type == 'ifg':
            ifg_names = list(self.stack.ifgs.keys())
            master_names = [ifg_key[:8] for ifg_key in ifg_names]
            slave_names = [ifg_key[9:] for ifg_key in ifg_names]

            ifg = []
            master = []
            slave = []
            coreg_master = []

            if slice:
                for ifg_name, master_name, slave_name in zip(ifg_names, master_names, slave_names):
                    for slice_name in slice_names:
                        if slice_name in self.stack.slcs[master_name].slice_data.keys() and \
                            slice_name in self.stack.slcs[slave_name].slice_data.keys():

                            ifg.append(self.stack.ifgs[ifg_name].slice_data[slice_name])
                            master.append(self.stack.slcs[master_name].slice_data[slice_name])
                            slave.append(self.stack.slcs[slave_name].slice_data[slice_name])
                            coreg_master.append(self.stack.slcs[coreg_master_name].slice_data[slice_name])
            else:
                for ifg_name, master_name, slave_name in zip(ifg_names, master_names, slave_names):
                    if concat_meta:
                        ifg.append(self.stack.ifgs[ifg_name])
                        master.append(self.stack.slcs[master_name])
                        slave.append(self.stack.slcs[slave_name])
                        coreg_master.append(self.stack.slcs[coreg_master_name])
                    else:
                        ifg.append(self.stack.ifgs[ifg_name].data)
                        master.append(self.stack.slcs[master_name].data)
                        slave.append(self.stack.slcs[slave_name].data)
                        coreg_master.append(self.stack.slcs[coreg_master_name].data)

            return ifg, master, slave, coreg_master

        elif data_type == 'coreg_master':

            if slice:
                coreg_master = [self.stack.slcs[coreg_master_name].slice_data[slice_name] for slice_name in slice_names]
            else:
                if concat_meta:
                    coreg_master = [self.stack.slcs[coreg_master_name]]
                else:
                    coreg_master = [self.stack.slcs[coreg_master_name].data]

            return coreg_master

        elif data_type == 'coreg_slave':

            slave_names = [slave_name for slave_name in list(self.stack.slcs.keys())]
            slave = []
            master = []

            if slice:
                for slave_name in slave_names:
                    for slice_name in slice_names:
                        if slice_name in self.stack.slcs[slave_name].slice_data.keys():
                            slave.append(self.stack.slcs[slave_name].slice_data[slice_name])
                            master.append(self.stack.slcs[coreg_master_name].slice_data[slice_name])
            else:
                for slave_name in slave_names:
                    if concat_meta:
                        slave.append(self.stack.slcs[slave_name])
                        master.append(self.stack.slcs[coreg_master_name])
                    else:
                        slave.append(self.stack.slcs[slave_name].data)
                        master.append(self.stack.slcs[coreg_master_name].data)


            return slave, master

    def save_data(self, meta_data, update_index=True):
        """
        Save the output data to the stack. This is done by indexing on the data folder as a unique id.

        :param meta_data:
        :return:
        """

        print('Working on this!')

    def download_sentinel_data(self, start_date, end_date, track, polarisation,
                              shapefile, radar_database_folder, orbit_folder,
                              ESA_username, ESA_password, ASF_username, ASF_password):
        """
        Creation of datastack of Sentinel-1 images including the orbits.

        :param start_date:
        :param end_date:
        :param track:
        :param polarisation:
        :param shapefile:
        :param radar_database_folder:
        :param orbit_folder:
        :param ESA_username:
        :param ESA_password:
        :param ASF_username:
        :param ASF_password:
        :return:
        """

        if polarisation is str:
            polarisation = [polarisation]

        # Download data and orbit
        for pol in polarisation:
            download_data = DownloadSentinel(start_date, end_date, shapefile, track, polarisation=pol)
            download_data.sentinel_available(ESA_username, ESA_password)
            download_data.sentinel_download_ASF(radar_database_folder, ASF_username, ASF_password)
            download_data.sentinel_check_validity(radar_database_folder, ESA_username, ESA_password)

        # Orbits
        precise_folder = os.path.join(orbit_folder, 'precise')
        download_orbit = DownloadSentinelOrbit(start_date, end_date, precise_folder)
        download_orbit.download_orbits()

    def create_sentinel_stack(self, start_date, end_date, master_date, track, polarisation,
                              shapefile, radar_database_folder, orbit_folder, stack_folder,
                              mode='IW', product_type='SLC'):
        """
        Creation of datastack of Sentinel-1 images including the orbits.

        :param start_date:
        :param end_date:
        :param master_date:
        :param track:
        :param polarisation:
        :param shapefile:
        :param radar_database_folder:
        :param orbit_folder:
        :param stack_folder:
        :param mode:
        :param product_type:
        :return:
        """

        # Number of cores
        cores = 6

        if polarisation is str:
            polarisation = [polarisation]

        # Prepare processing
        for pol in polarisation:
            self.stack = SentinelStack(stack_folder)
            self.stack.read_from_database(radar_database_folder, shapefile, track, orbit_folder, start_date, end_date,
                                          master_date, mode, product_type, pol, cores=cores)
            self.read_stack(stack_folder, start_date, end_date)

    def read_stack(self, stack_folder, start_date, end_date):
        """
        Read information of stack

        :param stack_folder:
        :param start_date:
        :param end_date:
        :return:
        """
        
        self.start_date = start_date
        self.end_date = end_date
        self.stack_folder = stack_folder
        
        self.stack = Stack(stack_folder)
        self.stack.read_master_slice_list()
        self.stack.read_stack(start_date, end_date)
    
    def reload_stack(self):
        """
        Reload stack
        
        :return: 
        """
    
        self.stack = Stack(self.stack_folder)
        self.stack.read_master_slice_list()
        self.stack.read_stack(self.start_date, self.end_date)
    
    def create_dem_coordinates(self, dem_type, lon_resolution=3):
        """
        Create the coordinate system for the dem

        :param dem_type:
        :param lon_resolution:
        :return:
        """

        if dem_type in ['SRTM1', 'SRTM3']:
            self.dem_coor = ImportDem.create_dem_coor(dem_type)
        else:
            self.dem_coor = ImportDem.create_dem_coor(dem_type, lon_resolution)

        self.dem_type = dem_type
        self.lon_resolution = lon_resolution

    def create_ml_coordinates(self, coor_type, multilook=[1,1], oversample=[1,1], shape=[0,0],
                              dlat=0.001, dlon=0.001, lat0=-90, lon0=180,
                              dx=1, dy=1, x0=0, y0=0, projection_string='', projection_type='',
                              reference_processing_step='crop'):
        """
        Create the coordinate system for multilooking. This can either be in radar coordinates, geographic or projected.

        :return:
        """

        self.full_ml_coor = CoordinateSystem()

        if coor_type == 'radar_grid':
            self.full_ml_coor.create_radar_coordinates(multilook=multilook, oversample=oversample, shape=shape)
            # TODO add readfiles/orbits
        elif coor_type == 'geographic':
            self.full_ml_coor.create_geographic(dlat, dlon, shape=shape, lon0=lon0, lat0=lat0)
        elif coor_type == 'projection':
            self.full_ml_coor.create_projection(dx, dy, projection_type=projection_type, proj4_str=projection_string, x0=x0, y0=y0)

        coreg_image = self.get_data('coreg_master', slice=False, concat_meta=False)[0]  # type: ImageProcessingData
        orbit = coreg_image.find_best_orbit()
        readfile = coreg_image.readfiles['original']
        self.full_radar_coor.load_orbit(orbit)

        # Define the full image multilooked image.
        if shape == [0, 0] or shape == '':
            new_coor = CoorNewExtend(self.full_radar_coor, self.full_ml_coor)
            self.full_ml_coor = new_coor.out_coor

        self.full_ml_coor.load_orbit(orbit)
        self.full_ml_coor.load_readfile(readfile)

    def create_radar_coordinates(self):
        """
        Create the radar coordinate system.

        :return:
        """

        self.radar_coor = CoordinateSystem()
        self.radar_coor.create_radar_coordinates()

        coreg_image = self.get_data('coreg_master', slice=False, concat_meta=False)[0]  # type: ImageProcessingData
        readfile = coreg_image.readfiles['original']

        self.full_radar_coor = copy.deepcopy(self.radar_coor)
        self.full_radar_coor.load_readfile(readfile)
        self.full_radar_coor.shape = readfile.size

    def create_ifg_network(self, image_baselines=[], network_type='temp_baseline',
                                     temporal_baseline=60, temporal_no=3, spatial_baseline=2000):
        """
        Create a network of interferograms.

        :return:
        """

        self.stack.create_interferogram_network(image_baselines, network_type, temporal_baseline, temporal_no,
                                                spatial_baseline)

    def download_external_dem(self, dem_folder, dem_type, NASA_username='', NASA_password='', DLR_username='', DLR_password='',
                              lon_resolution=3, buffer=1, rounding=1):
        """

        :param dem_folder:
        :param dem_type:
        :param DLR_username:
        :param DLR_password:
        :param NASA_username:
        :param NASA_password:
        :param lon_resolution:
        :return:
        """

        self.dem_buffer = buffer
        self.dem_rounding = rounding

        if dem_type == 'SRTM1':
            self.stack.download_SRTM_dem(dem_folder, NASA_username, NASA_password, buffer=buffer, rounding=rounding,
                                            srtm_type='SRTM1')

        elif dem_type == 'SRTM3':
            self.stack.download_SRTM_dem(dem_folder, NASA_username, NASA_password, buffer=buffer, rounding=rounding,
                                            srtm_type='SRTM3')
        elif dem_type == 'Tandem_X':
            self.stack.download_Tandem_X_dem(dem_folder, DLR_username, DLR_password, buffer=buffer, rounding=rounding,
                                           lon_resolution=lon_resolution)

    def geocoding(self, dem_folder='', dem_type='', dem_buffer='', dem_rounding='', lon_resolution=''):
        """
        :param dem_folder:
        :param dem_type:
        :param dem_buffer:
        :param dem_rounding:
        :param lon_resolution:
        :return:
        """

        # Load parameters to import DEM
        if dem_folder:
            self.dem_folder = dem_folder
        if dem_type:
            self.dem_type = dem_type
        if dem_buffer:
            self.dem_buffer = dem_buffer
        if dem_rounding:
            self.dem_rounding = dem_rounding
        if lon_resolution:
            self.lon_resolution = lon_resolution

        # Create the first multiprocessing pipeline.
        self.reload_stack()
        coreg_slices = self.get_data('coreg_master', slice=True)

        dem_pipeline = Pipeline(pixel_no=5000000, processes=self.processes)
        dem_pipeline.add_processing_data(coreg_slices, 'coreg_master')
        dem_pipeline.add_processing_step(ImportDem(in_coor=self.radar_coor, coreg_master='coreg_master',
                                                   dem_type=self.dem_type, dem_folder=self.dem_folder,
                                                   buffer=self.dem_buffer, rounding=self.dem_rounding,
                                                   lon_resolution=self.lon_resolution), True)
        dem_pipeline.add_processing_step(InverseGeocode(in_coor=self.radar_coor, out_coor=self.dem_coor,
                                                        coreg_master='coreg_master', dem_type=self.dem_type), True)
        dem_pipeline()

        # Then create the radar DEM, geocoding, incidence angles for the master grid
        self.reload_stack()
        coreg_slices = self.get_data('coreg_master', slice=True)

        geocode_pipeline = Pipeline(pixel_no=3000000, processes=self.processes)
        geocode_pipeline.add_processing_data(coreg_slices, 'coreg_master')
        geocode_pipeline.add_processing_step(ResampleDem(out_coor=self.radar_coor, in_coor=self.dem_coor,
                                                         coreg_master='coreg_master'), True)
        geocode_pipeline.add_processing_step(Geocode(out_coor=self.radar_coor, coreg_master='coreg_master'), True)
        geocode_pipeline.add_processing_step(RadarRayAngles(out_coor=self.radar_coor, coreg_master='coreg_master'), True)
        geocode_pipeline()

        self.reload_stack()
        coreg_image = self.get_data('coreg_master', slice=False, concat_meta=True)[0]

        # Finally concatenate bursts
        coreg_image.create_concatenate_image(process='dem', file_type='dem', coor=self.dem_coor, replace=True)
        coreg_image.create_concatenate_image(process='dem', file_type='dem', coor=self.radar_coor, transition_type='cut_off')
        coreg_image.create_concatenate_image(process='geocode', file_type='lat', coor=self.radar_coor, transition_type='cut_off')
        coreg_image.create_concatenate_image(process='geocode', file_type='lon', coor=self.radar_coor, transition_type='cut_off')
        coreg_image.create_concatenate_image(process='radar_ray_angles', file_type='incidence_angle', coor=self.radar_coor, transition_type='cut_off')
        coreg_image.create_concatenate_image(process='radar_ray_angles', file_type='off_nadir_angle', coor=self.radar_coor, transition_type='cut_off')

    def geometric_coregistration_resampling(self, polarisation):
        """
        Coregistration and resampling based on topography only.

        :param polarisation: Used polarisation(s)
        :return:
        """

        if polarisation is str:
            polarisation = [polarisation]

        # Allow the processing of two polarisation at the same time.
        for pol in polarisation:
            self.reload_stack()
            [slave_slices, coreg_slices] = self.get_data('coreg_slave', slice=True)

            resampling_pipeline = Pipeline(pixel_no=5000000, processes=self.processes)
            resampling_pipeline.add_processing_data(coreg_slices, 'coreg_master')
            resampling_pipeline.add_processing_data(slave_slices, 'slave')
            resampling_pipeline.add_processing_step(
                GeometricCoregistration(out_coor=self.radar_coor, in_coor=self.radar_coor, coreg_master='coreg_master',
                                        slave='slave'), False)

            resampling_pipeline.add_processing_step(DerampResampleRadarGrid(out_coor=self.radar_coor, in_coor=self.radar_coor, polarisation=pol, slave='slave'), False)
            resampling_pipeline.add_processing_step(Reramp(out_coor=self.radar_coor, polarisation=pol, slave='slave'), False)
            resampling_pipeline.add_processing_step(EarthTopoPhase(out_coor=self.radar_coor, polarisation=pol, slave='slave'), True)
            resampling_pipeline()

        # Concatenate the images
        for pol in polarisation:
            self.reload_stack()
            [slave_images, coreg_image] = self.get_data('coreg_slave', slice=False, concat_meta=True)

            for slave in slave_images:
                slave.create_concatenate_image(process='earth_topo_phase', file_type='earth_topo_phase_corrected',
                                               coor=self.radar_coor, transition_type='cut_off', polarisation=pol)

            # Concatenate the files from the main folder
            coreg_image[0].create_concatenate_image(process='crop', file_type='crop', coor=self.radar_coor,
                                                    transition_type='cut_off', polarisation=pol)

    def prepare_multilooking_grid(self, polarisation):
        """
        Create a multilooking grid.

        :param polarisation: Polarisation used for processing
        :return:
        """

        self.reload_stack()
        coreg_image = self.get_data('coreg_master', slice=False, concat_meta=True)

        # Concatenate the files from the main folder if not already done
        coreg_image[0].create_concatenate_image(process='crop', file_type='crop', polarisation=polarisation, coor=self.radar_coor, transition_type='cut_off')

        coreg_image = self.get_data('coreg_master', slice=False)
        create_multilooking_grid = Pipeline(pixel_no=5000000, processes=self.processes)
        create_multilooking_grid.add_processing_data(coreg_image, 'coreg_master')
        create_multilooking_grid.add_processing_step(
            MultilookPrepare(in_polarisation=polarisation, in_coor=self.radar_coor, out_coor=self.full_ml_coor,
                             in_file_type='square_amplitude', in_process='square_amplitude',
                             slave='coreg_master', coreg_master='coreg_master'), True)
        create_multilooking_grid()

    def create_interferogram_multilooked(self, polarisation):
        """
        Create the interferograms using a multilooking in a geographic grid.

        :param polarisation:
        :return:
        """

        if polarisation is str:
            polarisation = [polarisation]

        # Then do the resampling
        for pol in polarisation:
            self.reload_stack()
            [ifgs, masters, slaves, coreg_master] = self.get_data('ifg', slice=False)

            create_multilooked_ifg = Pipeline(pixel_no=0, processes=self.processes)
            create_multilooked_ifg.add_processing_data(coreg_master, 'coreg_master')
            create_multilooked_ifg.add_processing_data(slaves, 'slave')
            create_multilooked_ifg.add_processing_data(masters, 'master')
            create_multilooked_ifg.add_processing_data(ifgs, 'ifg')
            create_multilooked_ifg.add_processing_step(
                InterferogramMultilook(polarisation=pol, in_coor=self.radar_coor, out_coor=self.full_ml_coor,
                                       slave='slave', coreg_master='coreg_master', ifg='ifg', master='master', batch_size=10000000), True, True)
            create_multilooked_ifg()

    def create_calibrated_amplitude_multilooked(self, polarisation):
        """
        Create a geographic grid for the calibrated amplitude images.

        :param polarisation:
        :return:
        """

        if polarisation is str:
            polarisation = [polarisation]

        for pol in polarisation:
            self.reload_stack()
            [coreg_slave, coreg_master] = self.get_data('coreg_slave')

            # First create the multilooked square amplitudes.
            create_multilooked_amp = Pipeline(pixel_no=0, processes=self.processes)
            create_multilooked_amp.add_processing_data(coreg_master, 'coreg_master')
            create_multilooked_amp.add_processing_data(coreg_slave, 'slave')
            create_multilooked_amp.add_processing_step(
                CalibratedAmplitudeMultilook(polarisation=pol, in_coor=self.radar_coor, out_coor=self.full_ml_coor,
                                       slave='slave', coreg_master='coreg_master', batch_size=10000000), True, True)
            create_multilooked_amp()
            create_multilooked_amp.save_processing_results()

            # Finally do the master image seperately
            amp_multilook = CalibratedAmplitudeMultilook(polarisation=pol, in_coor=self.radar_coor, out_coor=self.full_ml_coor,
                                                   slave=coreg_master[0], coreg_master=coreg_master[0], master_image=True, batch_size=10000000)
            amp_multilook()

    def create_coherence_multilooked(self, polarisation):
        """
        Create the coherence values for the interferogram. Make sure that you created the interferograms first!

        :param polarisation: Polarisation of dataset.
        :return:
        """

        if polarisation is str:
            polarisation = [polarisation]

        # First create the multilooked square amplitudes.
        for pol in polarisation:
            self.reload_stack()
            [coreg_slave, coreg_master] = self.get_data('coreg_slave')

            create_multilooked_amp = Pipeline(pixel_no=0, processes=self.processes)
            create_multilooked_amp.add_processing_data(coreg_master, 'coreg_master')
            create_multilooked_amp.add_processing_data(coreg_slave, 'slave')
            create_multilooked_amp.add_processing_step(
                SquareAmplitudeMultilook(polarisation=pol, in_coor=self.radar_coor, out_coor=self.full_ml_coor,
                                       slave='slave', coreg_master='coreg_master', batch_size=10000000), True, True)
            create_multilooked_amp()

            # Finally do the master image seperately
            amp_multilook = SquareAmplitudeMultilook(polarisation=pol, in_coor=self.radar_coor, out_coor=self.full_ml_coor,
                                                   slave=coreg_master[0], coreg_master=coreg_master[0], master_image=True, batch_size=10000000)
            amp_multilook()

        # After creation of the square amplitude images, we can create the coherence values themselves.
        # Create coherences for multilooked grid
        for pol in polarisation:
            self.reload_stack()
            [ifgs, masters, slaves, coreg_master] = self.get_data('ifg', slice=False)

            create_coherence = Pipeline(pixel_no=5000000, processes=self.processes)
            create_coherence.add_processing_data(coreg_master, 'coreg_master')
            create_coherence.add_processing_data(slaves, 'slave')
            create_coherence.add_processing_data(masters, 'master')
            create_coherence.add_processing_data(ifgs, 'ifg')
            create_coherence.add_processing_step(
                Coherence(polarisation=pol, out_coor=self.full_ml_coor, slave='slave', master='master', ifg='ifg'), True)
            create_coherence()

    def create_unwrapped_images(self, polarisation):
        """
        Create the unwrapped interferograms.

        :param polarisation:
        :return:
        """

        if polarisation is str:
            polarisation = [polarisation]

        for pol in polarisation:
            self.reload_stack()
            [ifgs, masters, slaves, coreg_master] = self.get_data('ifg', slice=False)

            create_unwrapped_image = Pipeline(pixel_no=0, processes=self.processes)
            create_unwrapped_image.add_processing_data(ifgs, 'ifg')
            create_unwrapped_image.add_processing_step(Unwrap(polarisation=pol, out_coor=self.full_ml_coor, ifg='ifg'), True)
            create_unwrapped_image()

    def create_geometry_mulitlooked(self, dem_folder='', dem_type='', dem_buffer='', dem_rounding='', lon_resolution=''):
        """
        Create the geometry for a geographic grid used for multilooking

        :return:
        """

        # Load parameters to import DEM
        if dem_folder:
            self.dem_folder = dem_folder
        if dem_type:
            self.dem_type = dem_type
        if dem_buffer:
            self.dem_buffer = dem_buffer
        if dem_rounding:
            self.dem_rounding = dem_rounding
        if lon_resolution:
            self.lon_resolution = lon_resolution

        # Create the first multiprocessing pipeline.
        self.reload_stack()
        coreg_slices = self.get_data('coreg_master', slice=False)

        dem_pipeline = Pipeline(pixel_no=5000000, processes=self.processes)
        dem_pipeline.add_processing_data(coreg_slices, 'coreg_master')
        dem_pipeline.add_processing_step(ImportDem(in_coor=self.full_ml_coor, coreg_master='coreg_master',
                                                   dem_type=self.dem_type, dem_folder=self.dem_folder,
                                                   buffer=self.dem_buffer, rounding=self.dem_rounding,
                                                   lon_resolution=self.lon_resolution), True)
        dem_pipeline()

        self.reload_stack()
        coreg_image = self.get_data('coreg_master', slice=False)
        create_resample_grid = Pipeline(pixel_no=5000000, processes=self.processes)
        create_resample_grid.add_processing_data(coreg_image, 'coreg_master')
        create_resample_grid.add_processing_step(
            ResamplePrepare(in_coor=self.dem_coor, out_coor=self.full_ml_coor,
                             in_file_type='dem', in_process='dem',
                             slave='coreg_master', coreg_master='coreg_master'), True)
        create_resample_grid()

        # Then create the radar DEM, geocoding, incidence angles for the master grid
        self.reload_stack()
        coreg_slices = self.get_data('coreg_master', slice=False)

        geocode_pipeline = Pipeline(pixel_no=3000000, processes=self.processes)
        geocode_pipeline.add_processing_data(coreg_slices, 'coreg_master')
        geocode_pipeline.add_processing_step(Resample(in_coor=self.dem_coor, out_coor=self.full_ml_coor,
                                                             in_file_type='dem', in_process='dem',
                                                             slave='coreg_master', coreg_master='coreg_master'), True)
        geocode_pipeline.add_processing_step(Geocode(out_coor=self.full_ml_coor, coreg_master='coreg_master'), True)
        geocode_pipeline.add_processing_step(RadarRayAngles(out_coor=self.full_ml_coor, coreg_master='coreg_master'), True)
        geocode_pipeline()

    def calculate_nesz_multilooked(self, nesz_main_dir):
        """
        Calculate the NESZ for the master image.
        Possible to add AASR and RASR (but that is actually not entirely correct)

        :return:
        """

        self.reload_stack()
        coreg_slices = self.get_data('coreg_master', slice=True)

        nesz_pipeline = Pipeline(pixel_no=10000000, processes=self.processes)
        nesz_pipeline.add_processing_data(coreg_slices, 'coreg_master')
        nesz_pipeline.add_processing_step(NeszHarmonySentinel(out_coor=self.radar_coor, coreg_master='coreg_master', nesz_main_dir=nesz_main_dir), True)
        nesz_pipeline()

        replace = True

        self.reload_stack()
        coreg_image = self.get_data('coreg_master', slice=False, concat_meta=True)[0]
        coreg_image.create_concatenate_image(process='nesz', file_type='nesz_harmony_ati', coor=self.radar_coor, transition_type='cut_off')
        coreg_image.create_concatenate_image(process='nesz', file_type='nesz_harmony_dual', coor=self.radar_coor, transition_type='cut_off')
        coreg_image.create_concatenate_image(process='nesz', file_type='nesz_sentinel', coor=self.radar_coor, transition_type='cut_off')

        self.reload_stack()
        coreg_image = self.get_data('coreg_master', slice=False, concat_meta=False)

        # First create the multilooked square amplitudes.
        create_multilooked_amp = Pipeline(pixel_no=0, processes=self.processes)
        create_multilooked_amp.add_processing_data(coreg_image, 'coreg_master')
        create_multilooked_amp.add_processing_step(
            Multilook(in_coor=self.radar_coor, out_coor=self.full_ml_coor, slave='coreg_master', coreg_master='coreg_master', batch_size=10000000,
                      in_file_type=['nesz_harmony_ati', 'nesz_harmony_dual', 'nesz_sentinel'], in_process='nesz'), True, True)
        create_multilooked_amp()
        create_multilooked_amp.save_processing_results()

    def create_output_tiffs_nesz(self):
        """
        Create the geotiff images

        :return:
        """

        self.reload_stack()
        geometry_datasets = self.stack.stack_data_iterator(['nesz'], coordinates=[self.full_ml_coor],
                                                           process_types=['nesz_harmony_ati', 'nesz_harmony_dual', 'nesz_sentinel'])[-1]
        coreg_master = self.get_data('coreg_master', slice=False)[0]
        readfile = coreg_master.readfiles['original']

        for geometry_dataset in geometry_datasets:  # type: ImageData
            geometry_dataset.coordinates.load_readfile(readfile)
            geometry_dataset.save_tiff(main_folder=True)
