# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.resample_irregular2regular import Irregular2Regular


class RadarDem(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', coor_in=[], coor_out=[], dem_type='SRTM1',
                 in_coor_types=[], in_processes=[], in_file_types=[], in_data_ids=[],
                 coreg_master=[], overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.

        :param CoordinateSystem coor_in: Coordinate system of the input grids.
        :param CoordinateSystem coor_out: Coordinate system of output grids, if not defined the same as coor_in
        :param list[str] in_coor_types: The coordinate types of the input grids. Sometimes some of these grids are
                different from the defined coor_in. Options are 'coor_in', 'coor_out' or anything else defined in the
                coordinate system input.

        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        """

        """
        First define the name and output types of this processing step.
        1. process_name > name of process
        2. file_types > name of process types that will be given as output
        3. data_types > names 
        """
        self.process_name = 'radar_dem'
        file_types = ['radar_dem']
        data_types = ['real4']

        """
        Then give the default input steps for the processing. The given input values will be overridden when other input
        values are given.
        """

        in_image_types = ['coreg_master', 'coreg_master', 'coreg_master']
        if len(in_coor_types) == 0:
            in_coor_types = ['coor_in', 'coor_in', 'coor_in']  # Same here but then for the coor_out and coordinate_systems
        if len(in_data_ids) == 0:
            in_data_ids = ['', '', '']
        in_polarisations = ['none', 'none', 'none']
        if len(in_processes) == 0:
            in_processes = ['create_dem', 'inverse_geocode', 'inverse_geocode']
        if len(in_file_types) == 0:
            in_file_types = ['dem', 'lines', 'pixels']

        in_type_names = ['dem', 'lines', 'pixels']

        # Initialize process step
        super(RadarDem, self).__init__(
                       process_name=self.process_name,
                       data_id=data_id,
                       file_types=file_types,
                       process_dtypes=data_types,
                       coor_in=coor_in,
                       coor_out=coor_out,
                       in_type_names=in_type_names,
                       in_coor_types=in_coor_types,
                       in_image_types=in_image_types,
                       in_processes=in_processes,
                       in_file_types=in_file_types,
                       in_polarisations=in_polarisations,
                       in_data_ids=in_data_ids,
                       coreg_master=coreg_master,
                       out_processing_image='coreg_master',
                       overwrite=overwrite)
        if self.process_finished:
            return

        # Define the irregular grid for input. This is needed if you want to calculate in blocks.
        self.in_irregular_grids = [self.in_images['lines'].disk['data'],
                                   self.in_images['pixels'].disk['data']]

    def process_calculations(self):
        """
        Calculate radar DEM using the

        :return:
        """

        resample = Irregular2Regular(self['lines'], self['pixels'], self['dem'], self.block_coor)
        self['radar_dem'] = resample()
