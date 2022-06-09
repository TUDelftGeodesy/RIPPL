# Import rippl template
from rippl.processing_templates.general_sentinel_1 import *

# Import processing functions
from rippl.processing_steps.NWP_delay_harmonie import HarmonieAPS
from rippl.processing_steps.NWP_delay_ecmwf import EcmwfAPS
from rippl.processing_steps.NWP_ifg_harmonie import InterferogramHarmonie
from rippl.processing_steps.NWP_ifg_ecmwf import InterferogramEcmwf


class APSTemplate(GeneralPipelines):
    
    def create_aps_coordinates(self, coor_type='geographic', multilook=[1,1], oversample=[1,1], shape=[0,0],
                              dlat=0.001, dlon=0.001, lat0=-90, lon0=-180, buffer=0, rounding=0,
                              dx=1, dy=1, x0=0, y0=0, projection_string='', projection_type='', standard_type=''):
        """
        Create the coordinate system for the aps interpolation. This should have a resolution of about 1-10 km, to 
        include all the data of the 

        :return:
        """

        if not standard_type:
            self.aps_coor = CoordinateSystem()

            if coor_type == 'radar_grid':
                self.aps_coor.create_radar_coordinates(multilook=multilook, oversample=oversample, shape=shape)
            elif coor_type == 'geographic':
                self.aps_coor.create_geographic(dlat, dlon, shape=shape, lon0=lon0, lat0=lat0)
            elif coor_type == 'projection':
                self.aps_coor.create_projection(dx, dy, projection_type=projection_type, proj4_str=projection_string, x0=x0, y0=y0)

        coreg_image = self.get_data('coreg_master', slice=False, concat_meta=False)[0]  # type: ImageProcessingData
        orbit = coreg_image.find_best_orbit()
        readfile = coreg_image.readfiles['original']
        self.full_radar_coor.load_orbit(orbit)
        self.full_radar_coor.create_radar_lines()

        # Define the full image multilooked image.
        if standard_type:
            new_coor = CoorNewExtend(self.full_radar_coor, standard_type, buffer=buffer, rounding=rounding,
                                     dx=dx, dy=dy, dlat=dlat, dlon=dlon)
            self.aps_coor = new_coor.out_coor
        elif shape == [0, 0] or shape == '':
            new_coor = CoorNewExtend(self.full_radar_coor, self.aps_coor, rounding=rounding, buffer=buffer)
            self.aps_coor = new_coor.out_coor

        self.aps_coor.load_orbit(orbit)
        self.aps_coor.load_readfile(readfile)

    def create_aps_geometry(self, block_orientation='lines'):
        """
        Create the geometry for a geographic grid used for multilooking

        :return:
        """

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
            ResamplePrepare(in_coor=self.dem_coor, out_coor=self.aps_coor,
                             in_file_type='dem', in_process='dem',
                             slave='coreg_master', coreg_master='coreg_master'), True)
        create_resample_grid()

        # Then create the radar DEM, geocoding, incidence angles for the master grid
        self.reload_stack()
        coreg_slices = self.get_data('coreg_master', slice=False)

        geocode_pipeline = Pipeline(pixel_no=3000000, processes=self.processes, block_orientation=block_orientation)
        geocode_pipeline.add_processing_data(coreg_slices, 'coreg_master')
        geocode_pipeline.add_processing_step(Resample(in_coor=self.dem_coor, out_coor=self.aps_coor,
                                                             in_file_type='dem', in_process='dem',
                                                             slave='coreg_master', coreg_master='coreg_master'), True)
        geocode_pipeline.add_processing_step(Geocode(out_coor=self.aps_coor, coreg_master='coreg_master'), True)
        geocode_pipeline.add_processing_step(RadarRayAngles(out_coor=self.aps_coor, coreg_master='coreg_master'), True)
        geocode_pipeline()
    
    def calculate_harmonie_aps(self, block_orientation='lines', time_delays=[], correct_time_delay=False):
        
        self.reload_stack()
        [slave_images, coreg_images] = self.get_data('coreg_slave', slice=False, include_coreg_master=True)

        aps_pipeline = Pipeline(pixel_no=5000000, processes=self.processes, block_orientation=block_orientation)
        aps_pipeline.add_processing_data(coreg_images, 'coreg_master')
        aps_pipeline.add_processing_data(slave_images, 'slave')
        aps_pipeline.add_processing_step(HarmonieAPS(out_coor=self.full_ml_coor, in_coor=self.aps_coor,
                                                     coreg_master='coreg_master',
                                                     slave='slave',
                                                     time_delays=time_delays,
                                                     correct_time_delay=correct_time_delay), True)
        aps_pipeline()

    def calculate_ifg_harmonie_aps(self, block_orientation='lines'):

        self.reload_stack()
        [ifgs, masters, slaves, coreg_master] = self.get_data('ifg', slice=False)

        create_harmonie_ifg = Pipeline(pixel_no=0, processes=self.processes, block_orientation=block_orientation)
        create_harmonie_ifg.add_processing_data(coreg_master, 'coreg_master')
        create_harmonie_ifg.add_processing_data(slaves, 'slave')
        create_harmonie_ifg.add_processing_data(masters, 'master')
        create_harmonie_ifg.add_processing_data(ifgs, 'ifg')
        create_harmonie_ifg.add_processing_step(
            InterferogramHarmonie(out_coor=self.full_ml_coor, slave='slave', coreg_master='coreg_master', ifg='ifg', master='master'), True, True)
        create_harmonie_ifg()

    def calculate_ecmwf_aps(self, block_orientation='lines', time_delays=[], correct_time_delay=False,
                            latlim=[-90, 90], lonlim=[-180, 180]):
        self.reload_stack()
        [slave_images, coreg_images] = self.get_data('coreg_slave', slice=False, include_coreg_master=True)

        aps_pipeline = Pipeline(pixel_no=5000000, processes=self.processes, block_orientation=block_orientation)
        aps_pipeline.add_processing_data(coreg_images, 'coreg_master')
        aps_pipeline.add_processing_data(slave_images, 'slave')
        aps_pipeline.add_processing_step(EcmwfAPS(out_coor=self.full_ml_coor, in_coor=self.aps_coor,
                                                 coreg_master='coreg_master', slave='slave',
                                                 latlim=latlim, lonlim=lonlim), True)
        aps_pipeline()

    def calculate_ifg_ecmwf_aps(self, block_orientation='lines'):

        self.reload_stack()
        [ifgs, masters, slaves, coreg_master] = self.get_data('ifg', slice=False)

        create_ecmwf_ifg = Pipeline(pixel_no=0, processes=self.processes, block_orientation=block_orientation)
        create_ecmwf_ifg.add_processing_data(coreg_master, 'coreg_master')
        create_ecmwf_ifg.add_processing_data(slaves, 'slave')
        create_ecmwf_ifg.add_processing_data(masters, 'master')
        create_ecmwf_ifg.add_processing_data(ifgs, 'ifg')
        create_ecmwf_ifg.add_processing_step(
            InterferogramEcmwf(out_coor=self.full_ml_coor, slave='slave', coreg_master='coreg_master', ifg='ifg', master='master'), True, True)
        create_ecmwf_ifg()

    def create_output_tiffs_aps_geometry(self, tiff_folder=''):
        """
        Create the geotiff images

        :return:
        """

        self.reload_stack()

        if not tiff_folder:
            tiff_folder = self.tiff_folder

        geometry_datasets = self.stack.stack_data_iterator(['radar_ray_angles', 'geocode', 'dem', 'baseline', 'height_to_phase'], coordinates=[self.aps_coor],
                                                           process_types=['lat', 'lon', 'incidence_angle', 'heading', 'dem', 'perpendicular_baseline', 'height_to_phase'], load_memmap=False)[-1]
        coreg_master = self.get_data('coreg_master', slice=False)[0]
        readfile = coreg_master.readfiles['original']

        for geometry_dataset in geometry_datasets:  # type: ImageData
            geometry_dataset.coordinates.load_readfile(readfile)
            geometry_dataset.save_tiff(tiff_folder=tiff_folder)

    def create_output_tiffs_harmonie_aps(self, tiff_folder=''):
        """
        Create the geotiff images

        :return:
        """

        self.reload_stack()

        if not tiff_folder:

            tiff_folder = self.tiff_folder

        harmonie_aps_datasets = self.stack.stack_data_iterator(['harmonie_aps', 'harmonie_interferogram'], coordinates=[self.full_ml_coor], load_memmap=False)[-1]

        for aps_dataset in harmonie_aps_datasets:  # type: ImageData
            aps_dataset.save_tiff(tiff_folder=tiff_folder)

    def create_output_tiffs_ecmwf_aps(self, tiff_folder=''):
        """
        Create the geotiff images

        :return:
        """

        self.reload_stack()

        if not tiff_folder:

            tiff_folder = self.tiff_folder

        ecmwf_aps_datasets = self.stack.stack_data_iterator(['ecmwf_aps', 'ecmwf_interferogram'], coordinates=[self.full_ml_coor], load_memmap=False)[-1]

        for aps_dataset in ecmwf_aps_datasets:  # type: ImageData
            aps_dataset.save_tiff(tiff_folder=tiff_folder)

    # Then the images for both harmonie and ecmwf data.
    def create_plots_ecmwf_aps(self, overwrite=False, factor=2, remove_sea=True):
        """

        :param overwrite:
        :return:
        """

        cmap = 'jet'
        ecmwf_datasets = self.stack.stack_data_iterator(['ecmwf_aps', 'ecmwf_interferogram'], coordinates=[self.full_ml_coor], ifg=True, load_memmap=False)[-1]
        for ecmwf_data in ecmwf_datasets:
            plot = PlotData(ecmwf_data, data_cmap=cmap, margins=0.1, overwrite=overwrite, factor=factor, remove_sea=remove_sea)
            succes = plot()
            if succes:
                plot.add_labels('ECWMF aps ' + os.path.basename(ecmwf_data.folder), 'Meters')
                plot.save_image()
                plot.close_plot()

    def create_plots_harmonie_aps(self, overwrite=False, factor=2, remove_sea=True):
        """

        :param overwrite:
        :return:
        """

        cmap = 'jet'
        harmonie_datasets = self.stack.stack_data_iterator(['harmonie_aps', 'harmonie_interferogram'], coordinates=[self.full_ml_coor], ifg=True, load_memmap=False)[-1]
        for harmonie_data in harmonie_datasets:

            plot = PlotData(harmonie_data, data_cmap=cmap, margins=0.1, overwrite=overwrite, factor=factor, remove_sea=remove_sea)
            succes = plot()
            if succes:
                plot.add_labels('Harmonie aps ' + os.path.basename(harmonie_data.folder), 'Meters')
                plot.save_image()
                plot.close_plot()
