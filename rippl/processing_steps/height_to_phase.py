# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates


class HeightToPhase(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='',
                 in_coor=[], radar_coor=[],
                 in_image_types=[], in_processes=[], in_file_types=[],
                 in_data_ids=[],
                 slave='slave', coreg_master='coreg_master', overwrite=False):

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

        """
        First define the name and output types of this processing step.
        1. process_name > name of process
        2. file_types > name of process types that will be given as output
        3. data_types > names 
        """

        self.process_name = 'height_to_phase'
        file_types = ['height_to_phase']
        data_types = ['real4']

        if radar_coor:
            if isinstance(radar_coor, CoordinateSystem):
                self.radar_coor = radar_coor
            else:
                raise TypeError('If a radar coordinate system is defined it should be an CoordinateSystem object')

        """
        Then give the default input steps for the processing. The given input values will be overridden when other input
        values are given.
        """

        if in_coor.grid_type == 'radar_coordinates':
            if len(in_image_types) == 0:
                in_image_types = ['coreg_master', 'coreg_master']
            in_coor_types = ['out_coor', 'out_coor']  # Same here but then for the out_coor and coordinate_systems
            if len(in_data_ids) == 0:
                in_data_ids = ['', '']
            in_polarisations = ['none', 'none']
            if len(in_processes) == 0:
                in_processes = ['baseline', 'radar_ray_angles']
            if len(in_file_types) == 0:
                in_file_types = ['perpendicular_baseline', 'incidence_angle']

            in_type_names = ['baseline', 'incidence_angle']
        else:
            if len(in_image_types) == 0:
                in_image_types = ['coreg_master', 'coreg_master', 'coreg_master', 'coreg_master', 'coreg_master']
            in_coor_types = ['out_coor', 'out_coor', 'out_coor']  # Same here but then for the out_coor and coordinate_systems
            if len(in_data_ids) == 0:
                in_data_ids = ['', '', '', '', '']
            in_polarisations = ['none', 'none', 'none', 'none', 'none']
            if len(in_processes) == 0:
                in_processes = ['baseline', 'radar_ray_angles', 'geocode', 'geocode', 'geocode']
            if len(in_file_types) == 0:
                in_file_types = ['perpendicular_baseline', 'incidence_angle', 'X', 'Y', 'Z']

            in_type_names = ['baseline', 'incidence_angle', 'X', 'Y', 'Z']

        super(HeightToPhase, self).__init__(
            process_name=self.process_name,
            data_id=data_id,
            file_types=file_types,
            process_dtypes=data_types,
            in_coor=in_coor,
            in_coor_types=in_coor_types,
            in_type_names=in_type_names,
            in_image_types=in_image_types,
            in_processes=in_processes,
            in_file_types=in_file_types,
            in_polarisations=in_polarisations,
            in_data_ids=in_data_ids,
            slave=slave,
            coreg_master=coreg_master,
            overwrite=overwrite)

    def process_calculations(self):
        """
        This step contains all the calculations that are done for this step. This is therefore the step that is most
        specific for every process definition.
        To run the full processing you have to initialize (__init__) and call (__call__) the object.

        To access the input data and save the output data you can the self.images dictionary. This contains all the
        memory data files. The keys are the same as the file_types and in_file_types.

        :return:
        """

        sol = 299792458

        if self.coordinate_systems['in_coor'].grid_type == 'radar_coordinates':

            self.coordinate_systems['in_coor'].create_radar_lines()
            R = (self.coordinate_systems['in_coor'].ra_time + self.coordinate_systems['in_coor'].interval_pixels * self.coordinate_systems['in_coor'].ra_step) / 2 * sol

            self['height_to_phase'] = self['baseline'] / (R[None, :] * np.sin(self['incidence'] / 180 * np.pi))

        else:
            xyz = np.vstack((np.ravel(self['X'])[None, :], np.ravel(self['Y'])[None, :], np.ravel(self['Z'])[None, :]))
            orbit_coordinates = OrbitCoordinates(self.radar_coor)
            lines, pixels = orbit_coordinates.xyz2lp(xyz)

            R = (self.radar_coor.ra_time + np.reshape(pixels, self.block_coor.shape) * self.radar_coor.ra_step) / 2 * sol

            self['height_to_phase'] = self['baseline'] / (R * np.sin(self['incidence'] / 180 * np.pi))
