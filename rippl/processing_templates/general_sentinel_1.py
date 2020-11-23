
from rippl.meta_data.stack import Stack
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData, ImageProcessingMeta
from rippl.meta_data.image_processing_concatenate import ImageConcatData
from rippl.SAR_sensors.sentinel.sentinel_stack import SentinelStack
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.SAR_sensors.sentinel.sentinel_download import DownloadSentinelOrbit, DownloadSentinel
from rippl.resampling.coor_new_extend import CoorNewExtend
from rippl.user_settings import UserSettings
from rippl.meta_data.plot_data import PlotData
from rippl.resampling.coor_concatenate import CoorConcatenate
import os
import numpy as np
import copy
import shutil
from shapely.geometry import Polygon
from shapely import speedups
speedups.disable()

from rippl.processing_steps.resample import Resample
from rippl.processing_steps.import_dem import ImportDem
from rippl.processing_steps.inverse_geocode import InverseGeocode
from rippl.processing_steps.resample_dem import ResampleDem
from rippl.processing_steps.geocode import Geocode
from rippl.processing_steps.geometric_coregistration import GeometricCoregistration
from rippl.processing_steps.calc_earth_topo_phase import CalcEarthTopoPhase
from rippl.processing_steps.square_amplitude import SquareAmplitude
from rippl.processing_steps.calc_reramp import CalcReramp
from rippl.processing_steps.correct_phases import CorrectPhases
from rippl.processing_steps.radar_ray_angles import RadarRayAngles
from rippl.processing_steps.baseline import Baseline
from rippl.processing_steps.height_to_phase import HeightToPhase
from rippl.processing_steps.coherence import Coherence
from rippl.processing_steps.deramp_resample_radar_grid import DerampResampleRadarGrid
from rippl.processing_steps.interferogram_multilook import InterferogramMultilook
from rippl.processing_steps.square_amplitude_multilook import SquareAmplitudeMultilook
from rippl.processing_steps.calibrated_amplitude_multilook import CalibratedAmplitudeMultilook
from rippl.processing_steps.multilook_prepare import MultilookPrepare
from rippl.processing_steps.resample_prepare import ResamplePrepare
from rippl.processing_steps.unwrap import Unwrap

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
        self.dem_buffer = None
        self.dem_coor = None
        self.dem_rounding = None
        self.dem_type = None
        self.dem_folder = None
        self.lon_resolution = 6

        self.stack_name = None
        self.stack_folder = None
        self.start_date = None
        self.end_date = None
        self.start_dates = None
        self.end_dates = None
        self.time_window = None
        self.dates = None
        self.tiff_folder = ''

    def get_data(self, data_type, slice=False, concat_meta=False, include_coreg_master=False):
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

            if include_coreg_master:
                slave_names = list(self.stack.slcs.keys())
            else:
                slave_names = [slave_name for slave_name in list(self.stack.slcs.keys()) if slave_name != coreg_master_name]
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

    def download_sentinel_data(self, start_date='', end_date='', date='', dates='', time_window='', start_dates='', end_dates=''
                               , track='', polarisation='', shapefile='', radar_database_folder=None, orbit_folder=None,
                              ESA_username=None, ESA_password=None, ASF_username=None, ASF_password=None, data=True, orbit=True,
                               source='ASF'):
        """
        Creation of data_stack of Sentinel-1 images including the orbits.

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

        if not isinstance(shapefile, Polygon):
            if not os.path.exists(shapefile):
                settings = UserSettings()
                settings.load_settings()
                shapefile = os.path.join(settings.GIS_database, shapefile)
            if not os.path.exists(shapefile):
                raise FileExistsError('Shapefile does not exist!')

        if data:
            if isinstance(polarisation, str):
                polarisation = [polarisation]

            # Download data and orbit
            for pol in polarisation:
                download_data = DownloadSentinel(start_date=start_date, end_date=end_date, end_dates=end_dates,
                                                 start_dates=start_dates, time_window=time_window, date=date, dates=dates,
                                                 shape=shapefile, track=track, polarisation=pol)
                if source == 'ASF':
                    download_data.sentinel_search_ASF(ASF_username, ASF_password)
                    download_data.sentinel_download_ASF(radar_database_folder, ASF_username, ASF_password)
                elif source == 'ESA':
                    download_data.sentinel_search_ESA(ESA_username, ESA_password)
                    download_data.sentinel_download_ESA(radar_database_folder, ESA_username, ESA_password)
                else:
                    print('Source should be ESA or ASF')

        # Orbits
        if orbit:

            settings = UserSettings()
            settings.load_settings()

            if not orbit_folder:
                orbit_folder = settings.orbit_database

            precise_folder = os.path.join(orbit_folder, settings.sar_sensor_name['sentinel1'], 'precise')
            if data:
                start_date = np.min(download_data.start_dates)
                end_date = np.max(download_data.end_dates)
            download_orbit = DownloadSentinelOrbit(start_date=start_date, end_date=end_date, precise_folder=precise_folder)
            download_orbit.download_orbits()

    def create_sentinel_stack(self, start_date='', end_date='', master_date='', track='', polarisation='VV', shapefile='',
                              date='', dates='', time_window='', start_dates='', end_dates='',
                              stack_name=None, radar_database_folder=None, orbit_folder=None,
                              stack_folder=None, mode='IW', product_type='SLC', cores=6, tiff_folder=''):
        """
        Creation of data_stack of Sentinel-1 images including the orbits.

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

        if isinstance(polarisation, str):
            polarisation = [polarisation]

        self.tiff_folder = tiff_folder

        # Prepare processing
        for pol in polarisation:
            self.stack = SentinelStack(data_stack_folder=stack_folder, data_stack_name=stack_name)
            self.stack.read_from_database(database_folder=radar_database_folder, shapefile=shapefile, track_no=track,
                                          orbit_folder=orbit_folder, start_date=start_date, end_date=end_date,
                                          master_date=master_date, mode=mode, product_type=product_type,
                                          polarisation=pol, cores=cores, date=date, dates=dates, time_window=time_window,
                                          start_dates=start_dates, end_dates=end_dates)
            self.read_stack(stack_folder=stack_folder, stack_name=stack_name, start_date=start_date, end_date=end_date,
                            date=date, dates=dates, time_window=time_window, start_dates=start_dates, end_dates=end_dates)
            self.stack.create_coverage_shp_kml_geojson()

    def read_stack(self, stack_folder='', stack_name='', start_date='', end_date='', start_dates='', end_dates='',
                   date='', dates='', time_window='', tiff_folder=''):
        """
        Read information of stack

        :param stack_folder:
        :param start_date:
        :param end_date:
        :return:
        """

        self.start_date = start_date
        self.end_date = end_date
        self.start_dates = start_dates
        self.end_dates = end_dates
        self.time_window = time_window
        self.dates = dates
        self.date = date
        self.stack_folder = stack_folder
        self.stack_name = stack_name
        self.tiff_folder = tiff_folder

        settings = UserSettings()
        settings.load_settings()

        self.stack = Stack(data_stack_folder=self.stack_folder, data_stack_name=self.stack_name, SAR_type=settings.sar_sensor_name['sentinel1'])
        self.stack.read_master_slice_list()
        self.stack.read_stack(start_date=start_date, end_date=end_date, start_dates=start_dates, end_dates=end_dates,
                              date=date, dates=dates, time_window=time_window)
    
    def reload_stack(self):
        """
        Reload stack
        
        :return: 
        """

        settings = UserSettings()
        settings.load_settings()

        self.stack = Stack(data_stack_folder=self.stack_folder, data_stack_name=self.stack_name, SAR_type=settings.sar_sensor_name['sentinel1'])
        self.stack.read_master_slice_list()
        self.stack.read_stack(start_date=self.start_date, end_date=self.end_date, start_dates=self.start_dates,
                              date=self.date, end_dates=self.end_dates, dates=self.dates, time_window=self.time_window)
    
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

    def create_ml_coordinates(self, coor_type='geographic', multilook=[1,1], oversample=[1,1], shape=[0,0],
                              dlat=0.001, dlon=0.001, lat0=-90, lon0=-180, buffer=0, rounding=0,
                              dx=1, dy=1, x0=0, y0=0, projection_string='', projection_type='', standard_type=''):
        """
        Create the coordinate system for multilooking. This can either be in radar coordinates, geographic or projected.

        :return:
        """

        if not standard_type:
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
        self.full_radar_coor.create_radar_lines()

        # Define the full image multilooked image.
        if standard_type:
            new_coor = CoorNewExtend(self.full_radar_coor, standard_type, buffer=buffer, rounding=rounding,
                                     dx=dx, dy=dy, dlat=dlat, dlon=dlon)
            self.full_ml_coor = new_coor.out_coor
        elif shape == [0, 0] or shape == '':
            new_coor = CoorNewExtend(self.full_radar_coor, self.full_ml_coor, rounding=rounding, buffer=buffer)
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

        coreg_image = self.get_data('coreg_master', slice=False, concat_meta=True)[0] # type: ImageConcatData
        coreg_image.load_full_meta()
        coreg_image.load_slice_meta()
        crop_coordinates = coreg_image.concat_image_data_iterator(['crop'], [self.radar_coor], slices=True)[3]

        concat_coors = CoorConcatenate(crop_coordinates)
        self.full_radar_coor = concat_coors.concat_coor

    def create_ifg_network(self, image_baselines=[], network_type='temp_baseline',
                                     temporal_baseline=60, temporal_no=3, spatial_baseline=2000):
        """
        Create a network of interferograms.

        :return:
        """

        self.stack.create_interferogram_network(image_baselines, network_type, temporal_baseline, temporal_no,
                                                spatial_baseline)

    def download_external_dem(self, dem_folder=None, dem_type='SRTM3', NASA_username=None, NASA_password=None, DLR_username=None, DLR_password=None,
                              lon_resolution=3, buffer=1, rounding=1, block_orientation='lines', n_processes=4):
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
                                            srtm_type='SRTM1', n_processes=n_processes)

        elif dem_type == 'SRTM3':
            self.stack.download_SRTM_dem(dem_folder, NASA_username, NASA_password, buffer=buffer, rounding=rounding,
                                            srtm_type='SRTM3', n_processes=n_processes)
        elif dem_type == 'TanDEM-X':
            self.stack.download_Tandem_X_dem(dem_folder, DLR_username, DLR_password, buffer=buffer, rounding=rounding,
                                           lon_resolution=lon_resolution, n_processes=n_processes)

    def geocoding(self, dem_folder=None, dem_type=None, dem_buffer=None, dem_rounding=None, lon_resolution=None, block_orientation='lines'):
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

        dem_pipeline = Pipeline(pixel_no=5000000, processes=self.processes, block_orientation=block_orientation)
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

        geocode_pipeline = Pipeline(pixel_no=3000000, processes=self.processes, block_orientation=block_orientation)
        geocode_pipeline.add_processing_data(coreg_slices, 'coreg_master')
        geocode_pipeline.add_processing_step(ResampleDem(out_coor=self.radar_coor, in_coor=self.dem_coor,
                                                         coreg_master='coreg_master'), True)
        geocode_pipeline.add_processing_step(Geocode(out_coor=self.radar_coor, coreg_master='coreg_master'), True)
        geocode_pipeline.add_processing_step(RadarRayAngles(out_coor=self.radar_coor, coreg_master='coreg_master',
                                                            heading=False, off_nadir_angle=False, azimuth_angle=False, incidence_angle=True), True,)
        geocode_pipeline()

        self.reload_stack()
        coreg_image = self.get_data('coreg_master', slice=False, concat_meta=True)[0]

        # Finally concatenate bursts
        coreg_image.create_concatenate_image(process='dem', file_type='dem', coor=self.dem_coor, replace=True, remove_input=False)
        coreg_image.create_concatenate_image(process='dem', file_type='dem', coor=self.radar_coor, transition_type='cut_off')
        coreg_image.create_concatenate_image(process='geocode', file_type='lat', coor=self.radar_coor, transition_type='cut_off', remove_input=False)
        coreg_image.create_concatenate_image(process='geocode', file_type='lon', coor=self.radar_coor, transition_type='cut_off', remove_input=False)
        coreg_image.create_concatenate_image(process='radar_ray_angles', file_type='incidence_angle', coor=self.radar_coor, transition_type='cut_off', remove_input=False)

    def geometric_coregistration_resampling(self, polarisation, output_phase_correction=False, block_orientation='lines',
                                            coreg_tmp_directory='', tmp_directory='', baselines=False, height_to_phase=False):
        """
        Coregistration and resampling based on topography only.

        :param polarisation: Used polarisation(s)
        :return:
        """

        if isinstance(polarisation, str):
            polarisation = [polarisation]

        # Allow the processing of two polarisation at the same time.
        self.reload_stack()
        [slave_slices, coreg_slices] = self.get_data('coreg_slave', slice=True, include_coreg_master=True)

        resampling_pipeline = Pipeline(pixel_no=5000000, processes=self.processes, block_orientation=block_orientation,
                                       coreg_tmp_directory=coreg_tmp_directory)
        resampling_pipeline.add_processing_data(coreg_slices, 'coreg_master')
        resampling_pipeline.add_processing_data(slave_slices, 'slave')
        resampling_pipeline.add_processing_step(GeometricCoregistration(out_coor=self.radar_coor,
                                                                        in_coor=self.radar_coor,
                                                                        coreg_master='coreg_master',
                                                                        slave='slave'), False)
        resampling_pipeline.add_processing_step(CalcEarthTopoPhase(out_coor=self.radar_coor, slave='slave'), False)
        resampling_pipeline.add_processing_step(CalcReramp(out_coor=self.radar_coor, slave='slave'), True)

        for pol in polarisation:
            resampling_pipeline.add_processing_step(DerampResampleRadarGrid(out_coor=self.radar_coor, in_coor=self.radar_coor,
                                                                            polarisation=pol, slave='slave'), False)
            resampling_pipeline.add_processing_step(CorrectPhases(out_coor=self.radar_coor, polarisation=pol, slave='slave'), True)

        if baselines or height_to_phase:
            if baselines:
                resampling_pipeline.add_processing_step(
                    Baseline(out_coor=self.radar_coor, slave='slave', coreg_master='coreg_master'), True)
            else:
                resampling_pipeline.add_processing_step(
                    Baseline(out_coor=self.radar_coor, slave='slave', coreg_master='coreg_master'), False)
            if height_to_phase:
                resampling_pipeline.add_processing_step(
                    HeightToPhase(out_coor=self.radar_coor, slave='slave', coreg_master='coreg_master'), True)

        resampling_pipeline()

        if coreg_tmp_directory:
            shutil.rmtree(coreg_tmp_directory)
            os.mkdir(coreg_tmp_directory)
        if tmp_directory:
            shutil.rmtree(tmp_directory)
            os.mkdir(tmp_directory)

        # Concatenate the images using parallel processing
        self.reload_stack()
        self.stack.create_concatenate_images(image_type='slc', process='calc_reramp', file_type='ramp', coor=self.full_radar_coor,
                                             transition_type='cut_off', remove_input=False, no_processes=self.processes, tmp_directory=tmp_directory)
        for pol in polarisation:
            self.stack.create_concatenate_images(image_type='slc', process='correct_phases', file_type='phase_corrected',
                                                 coor=self.full_radar_coor, transition_type='cut_off', remove_input=False,
                                                 no_processes=self.processes, polarisation=pol, tmp_directory=tmp_directory)

    def prepare_multilooking_grid(self, polarisation, block_orientation='lines'):
        """
        Create a multilooking grid.

        :param polarisation: Polarisation used for processing
        :return:
        """

        if isinstance(polarisation, list):
            polarisation = polarisation[0]

        self.reload_stack()
        coreg_image = self.get_data('coreg_master', slice=False, concat_meta=True)

        # Concatenate the files from the main folder if not already done
        coreg_image[0].create_concatenate_image(process='crop', file_type='crop', polarisation=polarisation, coor=self.full_radar_coor, transition_type='cut_off', remove_input=False)

        coreg_image = self.get_data('coreg_master', slice=False)
        create_multilooking_grid = Pipeline(pixel_no=5000000, processes=self.processes, block_orientation=block_orientation)
        create_multilooking_grid.add_processing_data(coreg_image, 'coreg_master')
        create_multilooking_grid.add_processing_step(
            MultilookPrepare(in_polarisation=polarisation, in_coor=self.radar_coor, out_coor=self.full_ml_coor,
                             in_file_type='square_amplitude', in_process='square_amplitude',
                             slave='coreg_master', coreg_master='coreg_master'), True)
        create_multilooking_grid()

    def create_interferogram_multilooked(self, polarisation, block_orientation='lines', tmp_directory='', coreg_tmp_directory=''):
        """
        Create the interferograms using a multilooking in a geographic grid.

        :param polarisation:
        :return:
        """

        if isinstance(polarisation, str):
            polarisation = [polarisation]

        # Then do the resampling
        for pol in polarisation:
            self.reload_stack()
            [ifgs, masters, slaves, coreg_master] = self.get_data('ifg', slice=False)

            create_multilooked_ifg = Pipeline(pixel_no=0, processes=self.processes, block_orientation=block_orientation,
                                              tmp_directory=tmp_directory, coreg_tmp_directory=coreg_tmp_directory)
            create_multilooked_ifg.add_processing_data(coreg_master, 'coreg_master')
            create_multilooked_ifg.add_processing_data(slaves, 'slave')
            create_multilooked_ifg.add_processing_data(masters, 'master')
            create_multilooked_ifg.add_processing_data(ifgs, 'ifg')
            create_multilooked_ifg.add_processing_step(
                InterferogramMultilook(polarisation=pol, in_coor=self.radar_coor, out_coor=self.full_ml_coor,
                                       slave='slave', coreg_master='coreg_master', ifg='ifg', master='master', batch_size=50000000), True, True)
            create_multilooked_ifg()

    def create_calibrated_amplitude_multilooked(self, polarisation, block_orientation='lines',
                                                tmp_directory='', coreg_tmp_directory=''):
        """
        Create a geographic grid for the calibrated amplitude images.

        :param polarisation:
        :return:
        """

        if isinstance(polarisation, str):
            polarisation = [polarisation]

        for pol in polarisation:
            self.reload_stack()
            [coreg_slave, coreg_master] = self.get_data('coreg_slave', include_coreg_master=True)

            # First create the multilooked square amplitudes.
            create_multilooked_amp = Pipeline(pixel_no=0, processes=self.processes, block_orientation=block_orientation,
                                              tmp_directory=tmp_directory, coreg_tmp_directory=coreg_tmp_directory)
            create_multilooked_amp.add_processing_data(coreg_master, 'coreg_master')
            create_multilooked_amp.add_processing_data(coreg_slave, 'slave')
            create_multilooked_amp.add_processing_step(
                CalibratedAmplitudeMultilook(polarisation=pol, in_coor=self.radar_coor, out_coor=self.full_ml_coor,
                                       slave='slave', coreg_master='coreg_master', batch_size=50000000, no_of_looks=True), True, True)
            create_multilooked_amp()
            create_multilooked_amp.save_processing_results()

    def create_calibrated_amplitude_approx_multilooked(self, polarisation, block_orientation='lines',
                                                       tmp_directory='', coreg_tmp_directory=''):
        """
        Create a geographic grid for the calibrated amplitude images.

        :param polarisation:
        :return:
        """

        if isinstance(polarisation, str):
            polarisation = [polarisation]

        for pol in polarisation:
            self.reload_stack()
            slcs = self.get_data('slc', concat_meta=True)

            for slc in slcs:
                slc.create_concatenate_image(process='crop', file_type='crop',
                                               coor=self.full_radar_coor, transition_type='cut_off', polarisation=pol, remove_input=False)

        for pol in polarisation:
            self.reload_stack()
            slcs = self.get_data('slc')
            # First create the multilooked square amplitudes.
            create_multilooked_amp = Pipeline(pixel_no=0, processes=self.processes, block_orientation=block_orientation,
                                              coreg_tmp_directory=coreg_tmp_directory, tmp_directory=tmp_directory)
            create_multilooked_amp.add_processing_data(slcs, 'slave')
            create_multilooked_amp.add_processing_data(slcs, 'coreg_master')
            create_multilooked_amp.add_processing_step(
                CalibratedAmplitudeMultilook(polarisation=pol, in_coor=self.radar_coor, out_coor=self.full_ml_coor,
                                       slave='slave', resampled=False, batch_size=50000000, no_line_pixel_input=True), True, True)
            create_multilooked_amp()

    def create_coherence_multilooked(self, polarisation, block_orientation='lines',
                                     tmp_directory='', coreg_tmp_directory=''):
        """
        Create the coherence values for the interferogram. Make sure that you created the interferograms first!

        :param polarisation: Polarisation of dataset.
        :return:
        """

        if isinstance(polarisation, str):
            polarisation = [polarisation]

        # First create the multilooked square amplitudes.
        for pol in polarisation:
            self.reload_stack()
            [coreg_slave, coreg_master] = self.get_data('coreg_slave', include_coreg_master=True)

            create_multilooked_amp = Pipeline(pixel_no=0, processes=self.processes, block_orientation=block_orientation,
                                              tmp_directory=tmp_directory, coreg_tmp_directory=coreg_tmp_directory)
            create_multilooked_amp.add_processing_data(coreg_master, 'coreg_master')
            create_multilooked_amp.add_processing_data(coreg_slave, 'slave')
            create_multilooked_amp.add_processing_step(
                SquareAmplitudeMultilook(polarisation=pol, in_coor=self.radar_coor, out_coor=self.full_ml_coor,
                                       slave='slave', coreg_master='coreg_master', batch_size=50000000), True, True)
            create_multilooked_amp()

        # After creation of the square amplitude images, we can create the coherence values themselves.
        # Create coherences for multilooked grid
        for pol in polarisation:
            self.reload_stack()
            [ifgs, masters, slaves, coreg_master] = self.get_data('ifg', slice=False)

            create_coherence = Pipeline(pixel_no=5000000, processes=self.processes, block_orientation=block_orientation)
            create_coherence.add_processing_data(coreg_master, 'coreg_master')
            create_coherence.add_processing_data(slaves, 'slave')
            create_coherence.add_processing_data(masters, 'master')
            create_coherence.add_processing_data(ifgs, 'ifg')
            create_coherence.add_processing_step(
                Coherence(polarisation=pol, out_coor=self.full_ml_coor, slave='slave', master='master', ifg='ifg'), True)
            create_coherence()

    def create_unwrapped_images(self, polarisation, block_orientation='lines'):
        """
        Create the unwrapped interferograms.

        :param polarisation:
        :return:
        """

        if isinstance(polarisation, str):
            polarisation = [polarisation]

        for pol in polarisation:
            self.reload_stack()
            [ifgs, masters, slaves, coreg_master] = self.get_data('ifg', slice=False)

            create_unwrapped_image = Pipeline(pixel_no=0, processes=self.processes, block_orientation=block_orientation)
            create_unwrapped_image.add_processing_data(ifgs, 'ifg')
            create_unwrapped_image.add_processing_step(Unwrap(polarisation=pol, out_coor=self.full_ml_coor, ifg='ifg'), True)
            create_unwrapped_image()

    def create_geometry_mulitlooked(self, dem_folder=None, dem_type=None, dem_buffer=None, dem_rounding=None, lon_resolution=None, block_orientation='lines',
                                    baselines=False, height_to_phase=False):
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

        dem_pipeline = Pipeline(pixel_no=5000000, processes=self.processes, block_orientation=block_orientation)
        dem_pipeline.add_processing_data(coreg_slices, 'coreg_master')
        dem_pipeline.add_processing_step(ImportDem(in_coor=self.radar_coor, coreg_master='coreg_master',
                                                   dem_type=self.dem_type, dem_folder=self.dem_folder,
                                                   buffer=self.dem_buffer, rounding=self.dem_rounding,
                                                   lon_resolution=self.lon_resolution), True)
        dem_pipeline()

        self.reload_stack()
        coreg_image = self.get_data('coreg_master', slice=False)
        create_resample_grid = Pipeline(pixel_no=5000000, processes=self.processes, block_orientation=block_orientation)
        create_resample_grid.add_processing_data(coreg_image, 'coreg_master')
        create_resample_grid.add_processing_step(
            ResamplePrepare(in_coor=self.dem_coor, out_coor=self.full_ml_coor,
                             in_file_type='dem', in_process='dem',
                             slave='coreg_master', coreg_master='coreg_master'), True)
        create_resample_grid()

        # Then create the radar DEM, geocoding, incidence angles for the master grid
        self.reload_stack()
        coreg_slices = self.get_data('coreg_master', slice=False)

        geocode_pipeline = Pipeline(pixel_no=3000000, processes=self.processes, block_orientation=block_orientation)
        geocode_pipeline.add_processing_data(coreg_slices, 'coreg_master')
        geocode_pipeline.add_processing_step(Resample(in_coor=self.dem_coor, out_coor=self.full_ml_coor,
                                                             in_file_type='dem', in_process='dem',
                                                             slave='coreg_master', coreg_master='coreg_master'), True)
        geocode_pipeline.add_processing_step(Geocode(out_coor=self.full_ml_coor, coreg_master='coreg_master'), True)
        geocode_pipeline.add_processing_step(RadarRayAngles(out_coor=self.full_ml_coor, coreg_master='coreg_master'), True)
        geocode_pipeline()

        # Finally create the baselines for the slave images.
        if baselines or height_to_phase:
            self.reload_stack()
            [coreg_slave, coreg_master] = self.get_data('coreg_slave', include_coreg_master=True)

            create_baselines = Pipeline(pixel_no=5000000, processes=self.processes, block_orientation=block_orientation)
            create_baselines.add_processing_data(coreg_master, 'coreg_master')
            create_baselines.add_processing_data(coreg_slave, 'slave')
            create_baselines.add_processing_step(
                GeometricCoregistration(out_coor=self.full_ml_coor, in_coor=self.radar_coor, slave='slave',
                                        coreg_master='coreg_master', coreg_crop=False), False)
            if baselines:
                create_baselines.add_processing_step(
                    Baseline(out_coor=self.full_ml_coor, slave='slave', coreg_master='coreg_master'), True)
            else:
                create_baselines.add_processing_step(
                    Baseline(out_coor=self.full_ml_coor, slave='slave', coreg_master='coreg_master'), False)
            if height_to_phase:
                create_baselines.add_processing_step(
                    HeightToPhase(out_coor=self.full_ml_coor, slave='slave', coreg_master='coreg_master'), True)

            create_baselines()

    def create_output_tiffs_amplitude(self, tiff_folder=''):
        """
        Create the geotiff images

        :return:
        """

        if not tiff_folder:
            tiff_folder = self.tiff_folder

        calibrated_amplitudes = self.stack.stack_data_iterator(['calibrated_amplitude'], [self.full_ml_coor], ifg=False, load_memmap=False)[-1]
        for calibrated_amplitude in calibrated_amplitudes:          # type: ImageData
            calibrated_amplitude.save_tiff(tiff_folder=tiff_folder)

    def create_output_tiffs_geometry(self, tiff_folder=''):
        """
        Create the geotiff images

        :return:
        """

        if not tiff_folder:
            tiff_folder = self.tiff_folder

        geometry_datasets = self.stack.stack_data_iterator(['radar_ray_angles', 'geocode', 'dem', 'baseline', 'height_to_phase'], coordinates=[self.full_ml_coor],
                                                           process_types=['lat', 'lon', 'incidence_angle', 'dem', 'perpendicular_baseline', 'height_to_phase'], load_memmap=False)[-1]
        coreg_master = self.get_data('coreg_master', slice=False)[0]
        readfile = coreg_master.readfiles['original']

        for geometry_dataset in geometry_datasets:  # type: ImageData
            geometry_dataset.coordinates.load_readfile(readfile)
            geometry_dataset.save_tiff(tiff_folder=tiff_folder)

    def create_output_tiffs_coherence_ifg(self, tiff_folder=''):
        """
        Creates the geotiffs of coherence and unwrapped values.

        :return:
        """

        if not tiff_folder:
            tiff_folder = self.tiff_folder

        # Save the resulting coherences
        coherences = self.stack.stack_data_iterator(['coherence'], [self.full_ml_coor], ifg=True, load_memmap=False)[-1]
        for coherence in coherences:          # type: ImageData
            coherence.save_tiff(tiff_folder=tiff_folder)

        ifgs = self.stack.stack_data_iterator(['interferogram'], [self.full_ml_coor], ifg=True, load_memmap=False)[-1]
        for ifg in ifgs:          # type: ImageData
            ifg.save_tiff(tiff_folder=tiff_folder)


    def create_plots_coherence(self, overwrite=False):
        """
        Create plots for the coherences

        :return:
        """

        cmap = 'Greys_r'
        coherences = self.stack.stack_data_iterator(['coherence'], [self.full_ml_coor], ifg=True, load_memmap=False)[-1]
        for coherence in coherences:
            plot = PlotData(coherence, data_cmap=cmap, margins=0.1, data_quantiles=[0.05, 0.95], overwrite=overwrite)
            succes = plot()
            if succes:
                plot.add_labels('Coherence ' + os.path.basename(coherence.folder), 'Coherence')
                plot.save_image()
                plot.close_plot()

    def create_plots_ifg(self, overwrite=False):
        """

        :param overwrite:
        :return:
        """

        cmap = 'jet'
        coherences = self.stack.stack_data_iterator(['coherence'], [self.full_ml_coor], ifg=True, load_memmap=False)[-1]
        ifgs = self.stack.stack_data_iterator(['interferogram'], [self.full_ml_coor], ifg=True, load_memmap=False)[-1]
        for ifg, coherence in zip(ifgs, coherences):
            plot = PlotData(ifg, data_cmap=cmap, margins=0.1, data_min_max=[-np.pi, np.pi], transparency_in=coherence,
                            transparency_scale='linear', complex_plot='phase', transparency_min_max=[0.1, 0.3], overwrite=overwrite)
            succes = plot()
            if succes:
                plot.add_labels('Interferogram ' + os.path.basename(ifg.folder), 'Radians')
                plot.save_image()
                plot.close_plot()

    def create_output_tiffs_unwrap(self, tiff_folder=''):
        """
        Creates geotiffs of unwrapped images.

        """

        if not tiff_folder:
            tiff_folder = self.tiff_folder

        # Save the resulting coherences
        unwrapped_images = self.stack.stack_data_iterator(['unwrap'], [self.full_ml_coor], ifg=True, load_memmap=False)[-1]
        for unwrapped in unwrapped_images:          # type: ImageData
            unwrapped.save_tiff(tiff_folder=tiff_folder)
