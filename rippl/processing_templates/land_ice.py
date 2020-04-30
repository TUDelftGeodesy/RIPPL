from rippl.processing_templates.general_sentinel_1 import GeneralPipelines
import gdal
from osgeo import osr
from osgeo import ogr
import os


class LandIce(GeneralPipelines):

    def create_output_tiffs_amplitude(self):
        """
        Create the geotiff images

        :return:
        """

        calibrated_amplitudes = self.stack.stack_data_iterator(['calibrated_amplitude'], [self.full_ml_coor], ifg=False)[-1]
        for calibrated_amplitude in calibrated_amplitudes:          # type: ImageData
            calibrated_amplitude.save_tiff(main_folder=True)

        geometry_datasets = self.stack.stack_data_iterator(['radar_ray_angles', 'geocode'], coordinates=[self.full_ml_coor],
                                                           process_types=['lat', 'lon', 'incidence_angle'])[-1]
        for geometry_dataset in geometry_datasets:                  # type: ImageData
            geometry_dataset.save_tiff(main_folder=True)

    def create_output_tiffs_geometry(self):
        """
        Create the geotiff images

        :return:
        """

        geometry_datasets = self.stack.stack_data_iterator(['radar_ray_angles', 'geocode', 'dem'], coordinates=[self.full_ml_coor],
                                                           process_types=['lat', 'lon', 'incidence_angle', 'dem'])[-1]
        coreg_master = self.get_data('coreg_master', slice=False)[0]
        readfile = coreg_master.readfiles['original']

        for geometry_dataset in geometry_datasets:  # type: ImageData
            geometry_dataset.coordinates.load_readfile(readfile)
            geometry_dataset.save_tiff(main_folder=True)

    def create_output_tiffs_coherence_ifg(self):
        """
        Creates the geotiffs of coherence and unwrapped values.

        :return:
        """

        # Save the resulting coherences
        coherences = self.stack.stack_data_iterator(['coherence'], [self.full_ml_coor], ifg=True)[-1]
        for coherence in coherences:          # type: ImageData
            coherence.save_tiff(main_folder=True)

        ifgs = self.stack.stack_data_iterator(['interferogram'], [self.full_ml_coor], ifg=True)[-1]
        for ifg in ifgs:          # type: ImageData
            ifg.save_tiff(main_folder=True)

    def create_output_tiffs_unwrap(self):
        """
        Creates geotiffs of unwrapped images.

        """

        # Save the resulting coherences
        unwrapped_images = self.stack.stack_data_iterator(['unwrap'], [self.full_ml_coor], ifg=True)[-1]
        for unwrapped in unwrapped_images:          # type: ImageData
            unwrapped.save_tiff(main_folder=True)

    def calc_ice_movement(self):
        """
        Calculates the movement of the ice using the resampled data as a starting point.

        :return:
        """

        print('Working on this')
