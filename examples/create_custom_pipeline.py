from rippl.processing_list import ProcessingList
from rippl.image_data import ImageData


class ProcessingPipeline():

    def __init__(self, coordinates, slaves, masters=[], coreg_masters=[], ifgs=[]):
        # Here we define the meta data, output coordinate system, number of lines.

        processes = ProcessingList()
        self.processes = processes.processing
        self.processing_inputs = processes.processing_inputs

        meta_data = slaves + masters + coreg_masters + ifgs

        for meta in meta_data:
            if not isinstance(ImageData, meta):
                print('All input metadatasets should be of type ImageData. Aborting...')
                return

        self.slaves = slaves
        self.masters = masters
        self.coreg_masters = coreg_masters
        self.ifgs = ifgs

        self.process_steps = []
        self.process_settings_names = []
        self.process_settings_data = []

    def add_processing_step(self, step, in_coordinates=[], settings=[]):
        # Add input coordinate systems and or other settings
        # It is not possible to change the coordinate system, as it should be fixed for the whole dataset.






    def add_output_file(self, step, file_types=[]):


    def run_pipeline(self, parallel=True, no_jobs=1, no_pixels=0):
        # Run the pipeline



class ConcatenatePipeline():

    def __init__(self, metas, coordinates):
        #


    def add_multilooking(self, settings, coordinates_in):


    def add_interferogram(self, slaves, coordinates_in):


