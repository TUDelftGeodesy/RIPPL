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


class Amplitude(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', coor_in=[],
                 in_processes=[], in_file_types=[], in_polarisations=[], in_data_ids=[],
                 slave=[], overwrite=False):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem coor_in: Coordinate system of the input grids.

        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_polarisations: For which polarisation is it done. Leave empty if not relevant
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        """

        """
        First define the name and output types of this processing step.
        1. process_name > name of process
        2. file_types > name of process types that will be given as output
        3. data_types > names 
        """

        self.process_name = 'amplitude'
        file_types = ['amplitude']
        data_types = ['real4']

        """
        Then give the default input steps for the processing. The given input values will be overridden when other input
        values are given.
        """

        in_image_types = ['slave']
        in_coor_types = ['coor_in']
        if len(in_data_ids) == 0:
            in_data_ids = ['none']
        if len(in_polarisations) == 0:
            in_polarisations = ['']
        if len(in_processes) == 0:
            in_processes = ['earth_topo_phase']
        if len(in_file_types) == 0:
            in_file_types = ['earth_topo_phase']

        in_type_names = ['input_data']

        # Initialize
        super(Amplitude, self).__init__(
                       process_name=self.process_name,
                       data_id=data_id,
                       polarisation=polarisation,
                       file_types=file_types,
                       process_dtypes=data_types,
                       coor_in=coor_in,
                       in_type_names=in_type_names,
                       in_image_types=in_image_types,
                       in_processes=in_processes,
                       in_file_types=in_file_types,
                       in_polarisations=in_polarisations,
                       in_data_ids=in_data_ids,
                       slave=slave,
                       overwrite=overwrite)

    def process_calculations(self):
        """
        Create an amplitude image. (Very basic operation)

        :return:
        """

        self['amplitude'] = np.abs(self['earth_topo_phase'])
