# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class Interferogram(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='',
                 coor_in=[],
                 in_image_types=[], in_coor_types=[], in_processes=[], in_file_types=[], in_polarisations=[],
                 in_data_ids=[],
                 slave=[], master=[], ifg=[]):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem coor_in: Coordinate system of the input grids.
        :param CoordinateSystem coor_out: Coordinate system of output grids, if not defined the same as coor_in
        :param list[str] in_coor_types: The coordinate types of the input grids. Sometimes some of these grids are
                different from the defined coor_in. Options are 'coor_in', 'coor_out' or anything else defined in the
                coordinate system input.
        :param dict[CoordinateSystem] coordinate_systems: Here the alternative input coordinate systems can be defined.
                Only used in very specific cases.

        :param list[str] in_image_types: The type of the input ImageProcessingData objects (e.g. slave/master/ifg etc.)
        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_polarisations: For which polarisation is it done. Leave empty if not relevant
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData master: Master image, generally used when creating interferograms
        :param ImageProcessingData ifg: Interferogram of a master/slave combination
        """

        """
        First define the name and output types of this processing step.
        1. process_name > name of process
        2. file_types > name of process types that will be given as output
        3. data_types > names 
        """

        self.process_name = 'interferogram'
        file_types = ['interferogram']
        data_types = ['complex_real4']

        """
        Then give the default input steps for the processing. The given input values will be overridden when other input
        values are given.
        """
        if len(in_image_types) == 0:
            in_image_types = ['slave', 'master']  # In this case only the slave is used, so the input variable master,
            # coreg_master, ifg and processing_images are not needed.
            # However, if you override the default values this could change.
        if len(in_coor_types) == 0:
            in_coor_types = ['coor_in', 'coor_in']  # Same here but then for the coor_out and coordinate_systems
        if len(in_data_ids) == 0:
            in_data_ids = ['none', 'none']
        if len(in_polarisations) == 0:
            in_polarisations = ['', '']
        if len(in_processes) == 0:
            in_processes = ['earth_topo_phase', 'earth_topo_phase']
        if len(in_file_types) == 0:
            in_file_types = ['earth_topo_phase', 'earth_topo_phase']

        in_type_names = ['master', 'slave']

        # Initialize
        super(Interferogram, self).__init__(
                       process_name=self.process_name,
                       data_id=data_id, polarisation=polarisation,
                       file_types=file_types,
                       process_dtypes=data_types,
                       coor_in=coor_in,
                       in_coor_types=in_coor_types,
                       in_image_types=in_image_types,
                       in_processes=in_processes,
                       in_file_types=in_file_types,
                       in_polarisations=in_polarisations,
                       in_data_ids=in_data_ids,
                       in_type_names=in_type_names,
                       slave=slave,
                       master=master,
                       ifg=ifg,
                       out_processing_image='ifg')

    def process_calculations(self):
        """
        This function creates an interferogram without additional multilooking.

        :return:
        """

        self['ifg'] = self['earth_topo_phase'] * np.conjugate(self['slave'])
