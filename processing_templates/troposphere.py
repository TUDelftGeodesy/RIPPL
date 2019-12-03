"""
Code to calculate atmospheric delays

"""

from rippl.processing_templates.general import GeneralPipelines
import numpy as np
import copy

from rippl.processing_steps.import_dem import ImportDem
from rippl.processing_steps.inverse_geocode import InverseGeocode
from rippl.processing_steps.resample_dem import ResampleDem
from rippl.processing_steps.geocode import Geocode
from rippl.processing_steps.radar_ray_angles import RadarRayAngles
from rippl.pipeline import Pipeline


class Troposphere(GeneralPipelines):

    def create_output_tiffs_geometry(self):
        """
        Create the geotiff images

        :return:
        """

        geometry_datasets = self.stack.stack_data_iterator(['radar_ray_angles', 'geocode'], coordinates=[self.ml_coor],
                                                           process_types=['lat', 'lon', 'incidence_angle'])[-1]
        for geometry_dataset in geometry_datasets:  # type: ImageData
            geometry_dataset.save_tiff(main_folder=True)

    def create_output_tiffs_coherence_unwrap(self):
        """
        Creates the geotiffs of coherence and unwrapped values.

        :return:
        """

        # Save the resulting coherences
        coherences = self.stack.stack_data_iterator(['coherence'], [self.ml_radar_grid], ifg=True)[-1]
        for coherence in coherences:          # type: ImageData
            coherence.save_tiff(main_folder=True)

        ifgs = self.stack.stack_data_iterator(['interferogram'], [self.ml_radar_grid], ifg=True)[-1]
        for ifg in ifgs:          # type: ImageData
            ifg.save_tiff(main_folder=True)


    def calc_radar_multilooked_geometry(self, ml=[50, 200]):
        """
        Calculate the radar multilooked geometry.

        :param ml:
        :return:
        """

        self.ml_radar_grid = copy.copy(self.full_radar_coor)
        self.ml_radar_grid.multilook = ml
        self.ml_radar_grid.shape = [int(np.ceil(self.full_radar_coor.shape[0] / ml[0])),
                                    int(np.ceil(self.full_radar_coor.shape[1] / ml[1]))]
        
        # Now calculate the geometry.
        # Create the first multiprocessing pipeline.
        self.reload_stack()
        coreg_slices = self.get_data('coreg_master', slice=True)

        dem_pipeline = Pipeline(pixel_no=5000000, processes=self.processes)
        dem_pipeline.add_processing_data(coreg_slices, 'coreg_master')
        dem_pipeline.add_processing_step(ImportDem(in_coor=self.ml_radar_grid, coreg_master='coreg_master',
                                                   dem_type=self.dem_type, dem_folder=self.dem_folder,
                                                   buffer=self.dem_buffer, rounding=self.dem_rounding,
                                                   lon_resolution=self.lon_resolution), True)
        dem_pipeline.add_processing_step(InverseGeocode(in_coor=self.ml_radar_grid, out_coor=self.dem_coor,
                                                        coreg_master='coreg_master', dem_type=self.dem_type), True)
        dem_pipeline()

        # Then create the radar DEM, geocoding, incidence angles for the master grid
        self.reload_stack()
        coreg_slices = self.get_data('coreg_master', slice=True)

        geocode_pipeline = Pipeline(pixel_no=3000000, processes=self.processes)
        geocode_pipeline.add_processing_data(coreg_slices, 'coreg_master')
        geocode_pipeline.add_processing_step(ResampleDem(out_coor=self.ml_radar_grid, in_coor=self.dem_coor,
                                                         coreg_master='coreg_master'), True)
        geocode_pipeline.add_processing_step(Geocode(out_coor=self.ml_radar_grid, coreg_master='coreg_master'), True)
        geocode_pipeline.add_processing_step(RadarRayAngles(out_coor=self.ml_radar_grid, coreg_master='coreg_master'), True)
        geocode_pipeline()


    def calc_harmonie_delay(self, time_lags=[0]):
        """
        Calculate the tropospheric delays.

        :return:
        """





    def calc_ECMWF_delay(self, time_lags=[0]):
        """
        Calculate tropospheric delays

        :param time_lags:
        :return:
        """

