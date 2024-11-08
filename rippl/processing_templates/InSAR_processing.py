
import numpy as np
import logging

# Metadata general processing
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.stack import Stack
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.meta_data.plot_data import PlotData
from rippl.pipeline import Pipeline

# SAR data sources
from rippl.SAR_sensors.sentinel.sentinel_create_stack import SentinelStack

# Processing steps in SAR processing
from rippl.processing_steps.resample import Resample
from rippl.processing_steps.import_dem import ImportDem
from rippl.processing_steps.inverse_geocode import InverseGeocode
from rippl.processing_steps.resample_dem import ResampleDem
from rippl.processing_steps.geocode import Geocode
from rippl.processing_steps.geometric_coregistration import GeometricCoregistration
from rippl.processing_steps.radar_geometry import RadarGeometry
from rippl.processing_steps.baseline import Baseline
from rippl.processing_steps.coherence import Coherence
from rippl.processing_steps.interferogram import Interferogram
from rippl.processing_steps.intensity import Intensity
from rippl.processing_steps.calibrated_amplitude import CalibratedAmplitude
from rippl.processing_steps.deramp import Deramp
from rippl.processing_steps.resample_radar_grid import ResampleRadarGrid
from rippl.processing_steps.reramp import Reramp
from rippl.processing_steps.earth_topo_phase import EarthTopoPhase
from rippl.processing_steps.grid_transform import GridTransform
from rippl.processing_steps.unwrap import Unwrap
from rippl.processing_steps.multilook import Multilook


class InSAR_Processing(SentinelStack, Stack):

    def __init__(self, processes=6, stack_folder='', stack_name='', sensor='sentinel-1'):
        """
        Initialize the general S1 processing chain.

        """

        self.processes = processes
        self.dem_folder = None

        if sensor == 'sentinel-1':
            SentinelStack.__init__(self, data_stack_folder=stack_folder, data_stack_name=stack_name)
        else:
            Stack.__init__(self, data_stack_folder=stack_folder, data_stack_name=stack_name)

    def download_external_dem(self, dem_folder=None, dem_type='SRTM3', EarthData_username=None, EarthData_password=None,
                              DLR_username=None, DLR_password=None, chunk_orientation='lines', n_processes=4,
                              lon_resolution=None):
        """

        :param dem_folder:
        :param dem_type:
        :param DLR_username:
        :param DLR_password:
        :param EarthData_username:
        :param EarthData_password:
        :param lon_resolution:
        :return:
        """

        # First create the DEM coordinate system.
        coordinates = self.coordinates[dem_type]['full']
        if dem_type == 'SRTM1':
            self.download_SRTM_dem(dem_folder, EarthData_username, EarthData_password, srtm_type='SRTM1', n_processes=n_processes)
        elif dem_type == 'SRTM3':
            self.download_SRTM_dem(dem_folder, EarthData_username, EarthData_password, srtm_type='SRTM3', n_processes=n_processes)
        elif dem_type == 'TDM30':
            self.download_Tandem_X_dem(dem_folder, DLR_username, DLR_password, tandem_x_type='TDM30', n_processes=n_processes)
            if not lon_resolution:
                lon_resolution = 1
        elif dem_type == 'TDM90':
            self.download_Tandem_X_dem(dem_folder, DLR_username, DLR_password, tandem_x_type='TDM90', n_processes=n_processes)
            if not lon_resolution:
                lon_resolution = 3

        # Then create the DEM for the main reference image and for the reference image slices.
        reference_image = self.get_processing_data('reference_slc')[0]['reference_slc'][self.reference_date + '_full']
        reference_slices = self.get_processing_data('reference_slc', slice=True)[0]['reference_slc']

        dem_meta = self.coordinates[dem_type]['meta']
        full_image_run = ImportDem(in_coor=coordinates, reference_slc=reference_image, dem_type=dem_type,
                                   rounding=dem_meta['rounding'], buffer=dem_meta['buffer'], lon_resolution=lon_resolution,
                                   expected_min_height=dem_meta['min_height'], expected_max_height=dem_meta['max_height'])
        full_image_run()

        slice_names = list(reference_slices.keys())
        for slice_name in slice_names:
            # Slice DEM coor
            reference_coor = self.coordinates[dem_type]['slice' + slice_name.split('slice')[-1]]
            reference_slice = reference_slices[slice_name]
            slice_image_run = ImportDem(in_coor=reference_coor, reference_slc=reference_slice, dem_type=dem_type,
                                   rounding=dem_meta['rounding'], buffer=dem_meta['buffer'], lon_resolution=lon_resolution,
                                   expected_min_height=dem_meta['min_height'], expected_max_height=dem_meta['max_height'])
            slice_image_run()

    def geocode_resample(self, polarisation, dem_type='SRTM1', chunk_orientation='lines'):
        """
        This pipeline does the geocoding and resampling of the secondary slcs in one go. This is the most efficient way
        in case you are processing just one reference_slc/secondary_slc pair and saves only the resampled secondary slc
        to disk, which saves a lot of disk space.
        For larger stacks it is recommended to split these steps and run the geocode_calc_geomtry() and coregister_resample()
        function after each other. This stores the lat/lon/DEM and XYZ coordinates as intermediate step for the
        reference slc only. This geocoding data can then be used for every secondary_slc to apply coregistration and
        resample to the reference grid.

        """

        if isinstance(polarisation, str):
            polarisation = [polarisation]

        # Create the first multiprocessing pipeline.
        geocode_resample_pipeline = Pipeline(pixel_no=5000000, processes=self.processes, chunk_orientation=chunk_orientation,
                                    processing_type='secondary_slc', image_type='slice', stack=self)
        settings = self.coordinates[dem_type]['meta']
        geocode_resample_pipeline.add_processing_step(InverseGeocode(in_coor='reference_slc', out_coor=dem_type,
                                                            buffer=settings['buffer'], rounding=settings['rounding'],
                                                            min_height=settings['min_height'], max_height=settings['max_height'],
                                                            dem_type=dem_type), False)
        geocode_resample_pipeline.add_processing_step(ResampleDem(out_coor='reference_slc', in_coor=dem_type,
                                                        buffer=settings['buffer'], rounding=settings['rounding'],
                                                        min_height=settings['min_height'], max_height=settings['max_height'],
                                                        dem_type=dem_type), True)
        geocode_resample_pipeline.add_processing_step(Geocode(out_coor='reference_slc'), True)
        geocode_resample_pipeline.add_processing_step(
            GeometricCoregistration(out_coor='reference_slc', in_coor='secondary_slc'), False)

        for pol in polarisation:
            geocode_resample_pipeline.add_processing_step(Deramp(out_coor='secondary_slc', polarisation=pol), False)
            geocode_resample_pipeline.add_processing_step(ResampleRadarGrid(out_coor='reference_slc', in_coor='secondary_slc',
                                                    polarisation=pol), False)
            geocode_resample_pipeline.add_processing_step(Reramp(out_coor='reference_slc', polarisation=pol), False)
            geocode_resample_pipeline.add_processing_step(EarthTopoPhase(out_coor='reference_slc', polarisation=pol), True)
        geocode_resample_pipeline()

        self.fake_resampling_reference_slc(polarisation)

    def geocode_calc_geometry(self, dem_type='SRTM1', chunk_orientation='lines', only_DEM=False):
        """
        :param dem_type:
        :param chunk_orientation:
        :return:
        """

        # Create the first multiprocessing pipeline.
        geocode_pipeline = Pipeline(pixel_no=5000000, processes=self.processes, chunk_orientation=chunk_orientation,
                                    processing_type='reference_slc', image_type='slice', stack=self)
        settings = self.coordinates[dem_type]['meta']
        geocode_pipeline.add_processing_step(InverseGeocode(in_coor='reference_slc', out_coor=dem_type,
                                                            buffer=settings['buffer'], rounding=settings['rounding'],
                                                            min_height=settings['min_height'], max_height=settings['max_height'],
                                                            reference_slc='reference_slc', dem_type=dem_type), False)
        geocode_pipeline.add_processing_step(ResampleDem(out_coor='reference_slc', in_coor=dem_type,
                                                        buffer=settings['buffer'], rounding=settings['rounding'],
                                                        min_height=settings['min_height'], max_height=settings['max_height'],
                                                        dem_type=dem_type, reference_slc='reference_slc'), True)
        if not only_DEM:
            geocode_pipeline.add_processing_step(Geocode(out_coor='reference_slc', reference_slc='reference_slc'), True)
            geocode_pipeline.add_processing_step(RadarGeometry(out_coor='reference_slc', reference_slc='reference_slc',
                                                               heading=False, off_nadir_angle=False, azimuth_angle=False, incidence_angle=True), True)
        geocode_pipeline()

    def coregister_resample(self, polarisation, chunk_orientation='lines', only_DEM=False):
        """
        Coregistration and resampling based on topography only.

        :param polarisation: Used polarisation(s)
        :return:
        """

        if isinstance(polarisation, str):
            polarisation = [polarisation]

        # Allow the processing of two polarisation at the same time.
        resampling_pipeline = Pipeline(pixel_no=5000000, processes=self.processes, chunk_orientation=chunk_orientation,
                                       processing_type='secondary_slc', image_type='slice', stack=self)
        # This is not needed if the outputs from geocoding are already provided.
        resampling_pipeline.add_processing_step(GeometricCoregistration(out_coor='reference_slc',
                                                                        in_coor='secondary_slc',
                                                                        reference_slc='reference_slc',
                                                                        secondary_slc='secondary_slc',
                                                                        only_DEM=only_DEM), False)

        for pol in polarisation:
            resampling_pipeline.add_processing_step(Deramp(out_coor='secondary_slc',
                                                    polarisation=pol, secondary_slc='secondary_slc'), False)
            resampling_pipeline.add_processing_step(ResampleRadarGrid(out_coor='reference_slc', in_coor='secondary_slc',
                                                    polarisation=pol, secondary_slc='secondary_slc'), False)
            resampling_pipeline.add_processing_step(Reramp(out_coor='reference_slc', polarisation=pol,
                                                           secondary_slc='secondary_slc'), False)
            resampling_pipeline.add_processing_step(EarthTopoPhase(out_coor='reference_slc', polarisation=pol,
                                                                   secondary_slc='secondary_slc'), True)
        resampling_pipeline()

        self.fake_resampling_reference_slc(polarisation)


    def fake_resampling_reference_slc(self, polarisation):
        """
        This function applies a fake resampling step for the reference slc. Because the reference slc does not have
        to be coregistered and resampled, as it should be resampled to its own grid, we add a fake resampling step that
        refers to the original cropped data for every slice as being resampled and corrected for flat earth and
        topographic phase. This allows us to use the same setup for creating interferograms for all interferometric
        combinations later on during processing.

        :param polarisation: list of polarisation values or individual polarisation (VV, VH, HH, HV)

        """

        if isinstance(polarisation, str):
            polarisation = [polarisation]

        # Fake resampling of reference SLC for all slices (This is not needed because no shifts are applied)
        [images, coors] = self.get_processing_data(data_type='reference_slc', slice=True)
        for pol in polarisation:
            for slice_key in list(images['reference_slc'].keys()):
                coor = coors['reference_slc'][slice_key]
                slice = images['reference_slc'][slice_key]
                resample_reference = ResampleRadarGrid(out_coor=coor, in_coor=coor,
                                                       secondary_slc=slice, polarisation=pol, overwrite=True)
                resample_reference.fake_processing(output_file_name='crop', output_file_type='complex_int')
                reramp_reference = Reramp(out_coor=coor, secondary_slc=slice, polarisation=pol, overwrite=True)
                reramp_reference.fake_processing(output_file_name='crop', output_file_type='complex_int')
                phase_correction_reference = EarthTopoPhase(out_coor=coor, secondary_slc=slice,
                                                            polarisation=pol, overwrite=True)
                phase_correction_reference.fake_processing(output_file_name='crop', output_file_type='complex_int')

    def calc_calibrated_amplitude(self, polarisation, ml_name, C2=False, pix_sar=15000000):
        """
        Create a geographic grid for the calibrated amplitude images.

        :param polarisation:
        :return:
        """

        if isinstance(polarisation, str):
            polarisation = [polarisation]

        ml_coor = self.coordinates[ml_name]['full']        # type: CoordinateSystem
        coor_meta = self.coordinates[ml_name]['meta']
        pix_num = self.get_multilooked_number_of_pixels(ml_coor, pix_sar=pix_sar)

        # First create the multilooked square amplitudes.
        create_ml_intensity = Pipeline(pixel_no=pix_num, processes=self.processes, chunk_orientation='chunks',
                                       processing_type='secondary_slc', image_type='slice', stack=self, include_reference=True)
        create_ml_intensity.add_processing_step(GridTransform(in_coor='reference_slc', out_coor=ml_name,
                             secondary_slc='secondary_slc', reference_slc='reference_slc',
                             in_coor_type='radar_coordinates', out_coor_type=ml_coor.grid_type), False)

        for pol in polarisation:
            create_ml_intensity.add_processing_step(Intensity(polarisation=pol, out_coor='reference_slc',
                                    secondary_slc='secondary_slc'), False)
            create_ml_intensity.add_processing_step(Multilook(in_coor='reference_slc', out_coor=ml_name,
                                        polarisation=pol, process='intensity', file_type='intensity',
                                        data_type='float32', secondary_slc='secondary_slc', reference_slc='reference_slc',
                                        buffer=coor_meta['buffer'], rounding=coor_meta['rounding'], number_of_samples=True,
                                        min_height=coor_meta['min_height'], max_height=coor_meta['max_height']), True)
        create_ml_intensity()

        # Create the concatenated image.
        self.reload_stack()
        images, coordinates = self.get_processing_data('slc', slice=False)
        for image_key in images['secondary_slc'].keys():
            for pol in polarisation:
                image = images['secondary_slc'][image_key]
                image.create_concatenate_image(process='intensity', file_type='intensity', coor=ml_coor,
                                               transition_type='full_weight', polarisation=pol)
                image.create_concatenate_image(process='intensity', file_type='number_of_samples', coor=ml_coor,
                                               transition_type='full_weight', polarisation=pol)

        # Process Intensity values to calibrated values.
        create_ml_amplitude = Pipeline(pixel_no=pix_sar, processes=self.processes, chunk_orientation='chunks',
                                       processing_type='secondary_slc', image_type='full', stack=self, include_reference=True)
        for pol in polarisation:
            create_ml_amplitude.add_processing_step(CalibratedAmplitude(polarisation=pol, out_coor=ml_name,
                                    secondary_slc='secondary_slc', reference_slc='reference_slc', multilooked=True), True)
        create_ml_amplitude()

    def calc_interferogram_coherence(self, polarisation, ml_name, intensity=False, pix_sar=15000000):
        """
        Create a multilooked image for the interferogram, coherence and amplitude

        """

        if isinstance(polarisation, str):
            polarisation = [polarisation]

        ml_coor = self.coordinates[ml_name]['full']        # type: CoordinateSystem
        coor_meta = self.coordinates[ml_name]['meta']
        pix_num = self.get_multilooked_number_of_pixels(ml_coor, pix_sar=pix_sar)

        # Load the data and calculate conversion between images
        self.reload_stack()

        # Initialize data
        create_ml_interferogram = Pipeline(pixel_no=pix_num, processes=self.processes, chunk_orientation='chunks',
                                       processing_type='ifg', image_type='slice', stack=self)
        create_ml_interferogram.add_processing_step(GridTransform(in_coor='reference_slc', out_coor=ml_name,
                             secondary_slc='secondary_slc', reference_slc='reference_slc',
                             in_coor_type='radar_coordinates', out_coor_type=ml_coor.grid_type), False)

        # Apply processing and multilooking for different operations and processing steps.
        for pol in polarisation:
            # Create interferogram
            create_ml_interferogram.add_processing_step(Interferogram(polarisation=pol, out_coor='reference_slc',
                                       secondary_slc='secondary_slc', ifg='ifg', primary_slc='primary_slc',), False)
            create_ml_interferogram.add_processing_step(Multilook(in_coor='reference_slc',
                                        out_coor=ml_name, polarisation=pol, process='interferogram',
                                        file_type='interferogram', data_type='complex64', secondary_slc='ifg',
                                        buffer=coor_meta['buffer'], rounding=coor_meta['rounding'],
                                        min_height=coor_meta['min_height'], max_height=coor_meta['max_height'],
                                        reference_slc='reference_slc', number_of_samples=False), True)
        create_ml_interferogram()

        # Concatenate the interferograms
        self.reload_stack()
        images, coordinates = self.get_processing_data('ifg', slice=False)
        for image_key in images['ifg'].keys():
            for pol in polarisation:
                image = images['ifg'][image_key]
                image.create_concatenate_image(process='interferogram', file_type='interferogram', coor=ml_coor,
                                           transition_type='full_weight', polarisation=pol)

        # Create coherence images (we assume that intensity is already available from calculating the amplitude)
        create_ml_coherence = Pipeline(pixel_no=pix_sar, processes=self.processes, chunk_orientation='chunks',
                                       processing_type='ifg', image_type='full', stack=self)
        for pol in polarisation:
            create_ml_coherence.add_processing_step(
                Coherence(polarisation=pol, out_coor=ml_name, secondary_slc='secondary_slc',
                          primary_slc='primary_slc', ifg='ifg'), True)
        create_ml_coherence()

    def unwrap(self, polarisation, ml_name, chunk_orientation='lines'):
        """
        Create the unwrapped interferograms.

        :param polarisation:
        :return:
        """
        if isinstance(polarisation, str):
            polarisation = [polarisation]

        self.reload_stack()
        create_unwrapped_image = Pipeline(pixel_no=0, processes=self.processes,
                                       processing_type='ifg', image_type='full', stack=self)
        for pol in polarisation:
            create_unwrapped_image.add_processing_step(
                Unwrap(polarisation=pol, out_coor=ml_name, ifg='ifg'), True)
        create_unwrapped_image()

    def geocode_calc_geometry_multilooked(self, ml_name='', dem_type='SRTM1', chunk_orientation='chunks',
                                          heading=False, azimuth_angle=False, incidence_angle=True, squint_angle=False,
                                          off_nadir_angle=False, baselines=False):
        """
        Create the geometry for a geographic grid used for multilooking

        :return:
        """

        ml_coor = self.coordinates[ml_name]['full']  # type: CoordinateSystem
        dem_coor = self.coordinates[dem_type]['full']

        # Create the first multiprocessing pipeline.
        create_ml_geocode = Pipeline(pixel_no=5000000, processes=self.processes, chunk_orientation='chunks',
                                       processing_type='reference_slc', image_type='full', stack=self)
        create_ml_geocode.add_processing_step(GridTransform(in_coor=dem_type, out_coor=ml_name, conversion_type='resample',
                                                           secondary_slc='reference_slc', reference_slc='reference_slc',
                                                            in_coor_type=dem_coor.grid_type, out_coor_type=ml_coor.grid_type), False)
        create_ml_geocode.add_processing_step(Resample(in_coor=dem_type, out_coor=ml_name, file_type='dem', process='dem',
                                                      secondary_slc='reference_slc', reference_slc='reference_slc'), True)
        create_ml_geocode.add_processing_step(Geocode(out_coor=ml_name, reference_slc='reference_slc'), True)
        create_ml_geocode.add_processing_step(RadarGeometry(out_coor=ml_name, reference_slc='reference_slc',
                                                           heading=False, off_nadir_angle=False, azimuth_angle=True,
                                                           incidence_angle=True, squint_angle=False), True)
        create_ml_geocode()

        # Finally create the baselines for the interograms
        if baselines:
            create_ml_baseline = Pipeline(pixel_no=5000000, processes=self.processes, chunk_orientation='chunks',
                                         processing_type='secondary_slc', image_type='full', stack=self)
            create_ml_baseline.add_processing_step(Baseline(out_coor=ml_name, secondary_slc='secondary_slc',
                                                          reference_slc='reference_slc'), True)
            create_ml_baseline()

    def plot_figures(self, process_name=[], variable_name=[], ml_name='', polarisation='', slices=False, ifg=True, slc=True,
                     overwrite=False, margins=0.1, quantiles=[0.001, 0.999], remove_sea=False, remove_land=False,
                     cmap='jet_r', title='', cbar_title='', factor=1,
                     dB_lims=[0, 0], coh_lims=[0, 0], linear_transparency=False):
        """
        Create plots for the coherences

        :return:
        """

        scaling = self.get_transparency(dB_lims, coh_lims, ml_name, linear_transparency)
        datasets = self.stack_data_iterator(processes=[process_name], coordinates=[self.coordinates[ml_name]['full']],
                                            process_types=[variable_name], load_memmap=False, polarisations=[polarisation],
                                            ifg=ifg, slc=slc, slices=slices)[-1]

        if len(datasets) == 0:
            logging.info('No datasets found for plotting. Aborting..')

        for dataset in datasets:
            if len(scaling) > 0:
                plot = PlotData(dataset, data_cmap=cmap, margins=margins, data_quantiles=quantiles,
                                transparency=scaling, complex_plot='phase', overwrite=overwrite,
                                remove_sea=remove_sea, remove_land=remove_land, factor=factor)
            else:
                plot = PlotData(dataset, data_cmap=cmap, margins=margins, data_quantiles=quantiles, overwrite=overwrite,
                                remove_sea=remove_sea, remove_land=remove_land, complex_plot='phase', factor=factor)
            succes = plot()
            if succes:
                plot.add_labels(title, cbar_title)
                plot.save_image()
                plot.close_plot()

    def get_transparency(self, dB_lims=[0, 0], coh_lims=[0, 0], ml_name='', linear_transparency=True):
        """

        """

        transparency = []

        for lims, process, variable in zip([dB_lims, coh_lims],
                                           ['calibrated_amplitude', 'coherence'],
                                           ['calibrated_amplitude_db', 'coherence']):
            if lims != [0, 0]:
                disk_data = self.stack_data_iterator([process], [self.coordinates[ml_name]['full']],
                                                     process_types=[variable], ifg=True,
                                                     load_memmap=False, polarisations=['VV'])[-1][-1]
                disk_data.load_disk_data()
                data = disk_data.disk2memory(disk_data.disk['data'], disk_data.dtype_disk)

                if len(transparency) == 0:
                    transparency = np.ones(data.shape)

                with np.errstate(invalid='ignore'):
                    if not linear_transparency:
                        transparency[data < lims[0]] = 0
                        transparency[data > lims[1]] = 0
                    else:
                        transparency[data < lims[0]] = 0
                        transparency[data > lims[1]] = transparency[data > lims[1]] * 1
                        in_between = (data < lims[1]) * (data > lims[0])
                        transparency[in_between] = transparency[in_between] * (data[in_between] - lims[0]) / (lims[1] - lims[0])

        return transparency

    def create_output_geotiffs(self, process_name=[], variable_name=[], ml_name=[], tiff_folder='', polarisation='',
                            slices=False, ifg=True, slc=True):
        """
        Creates geotiffs of unwrapped images.

        """

        self.reload_stack()

        if not isinstance(process_name, list):
            process_name = [process_name]
        if not isinstance(variable_name, list):
            variable_name = [variable_name]
        if not isinstance(ml_name, list):
            ml_name = [ml_name]

        coordinates = [self.coordinates[name]['full'] for name in ml_name]

        # Save the resulting coherences
        data_images = self.stack_data_iterator(processes=process_name, coordinates=coordinates,
                                                    process_types=variable_name, ifg=ifg, slc=slc, slices=slices)[-1]
        for data in data_images:          # type: ImageData
            if data.load_disk_data():
                data.save_tiff()
            else:
                logging.info(data.file_path + ' not found on disk, not able to create geotiff file.')

    @staticmethod
    def get_multilooked_number_of_pixels(ml_coor, pix_sar=5000000, sar_res=[5, 20]):
        """
        Get the number of pixels that should be selected from the multilooked grid to make sure that the radar
        coordinates chunk is not too large.

        """

        # Calculate the size of the chunks. We go for approximately 25x25 km cells. This gives a low number of grid
        # cells for the output multilooked grid, but it uses a large input radar grid, which is the limiting factor
        # for loading data in memory.

        if ml_coor.grid_type == 'geographic':
            # We calculate degrees latitude to distances using coarse approx
            mean_lat = ml_coor.lat0 + (ml_coor.first_line + (ml_coor.shape[0] / 2)) * ml_coor.dlat
            lat_dist = 111000
            lon_dist = np.cos(np.deg2rad(mean_lat)) * lat_dist

            chunk_size_degree = np.sqrt(pix_sar / ((lat_dist * lon_dist) / (sar_res[0] * sar_res[1])))
            pix_num = np.int32((chunk_size_degree / ml_coor.dlat) * (chunk_size_degree / ml_coor.dlon))
        elif ml_coor.grid_type == 'projection':
            chunk_size_m = np.sqrt(pix_sar * sar_res[0] * sar_res[1])
            pix_num = np.int32((chunk_size_m / ml_coor.dx) * (chunk_size_m / ml_coor.dy))
        elif ml_coor.grid_type == 'radar_coordinates':
            pix_num = pix_sar / (ml_coor.multilook[0] * ml_coor.multilook[1])

        return pix_num
