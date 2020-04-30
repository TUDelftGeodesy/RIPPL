from rippl.processing_templates.general_sentinel_1 import GeneralPipelines


class Oceans(GeneralPipelines):

    def create_output_tiffs(self):
        """
        Create the geotiff images

        :return:
        """

        calibrated_amplitudes = self.stack.stack_data_iterator(['calibrated_amplitude'], [self.ml_coor], ifg=False)[-1]
        for calibrated_amplitude in calibrated_amplitudes:          # type: ImageData
            calibrated_amplitude.save_tiff(main_folder=True)

        geometry_datasets = self.stack.stack_data_iterator(['radar_ray_angles', 'geocode'], coordinates=[self.ml_coor],
                                                           process_types=['lat', 'lon', 'incidence_angle'])[-1]
        for geometry_dataset in geometry_datasets:                  # type: ImageData
            geometry_dataset.save_tiff(main_folder=True)
