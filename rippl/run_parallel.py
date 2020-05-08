# Function to run parallel package. For further details inspect the pipeline class.
from rippl.meta_data.process import Process
from rippl.meta_data.multilook_process import MultilookProcess
from rippl.meta_data.image_processing_data import ImageProcessingData


def run_parallel(dat):
    """
    This function runs a parallel package for image processing.

    :param dict[list[Process] or list[bool]] dat:

    :return:
    """

    # Get all processing image datasets
    processing_images = []
    image_keys = []

    # Load data as memmaps
    for process in dat['processes']:
        for key in process.processing_images.keys():
            if key not in image_keys:
                processing_images.append(process.processing_images[key])
                image_keys.append(key)

    for processing_image in processing_images:      # type: ImageProcessingData
        processing_image.load_memmap_files()

    # Loop over all processes.
    for process, save, memory_in in zip(dat['processes'], dat['save_processes'], dat['memory_in']):
        try:
            # Load needed input shapes
            process.load_coordinate_system_sizes(find_out_coor=False)
            # First init the process inputs and outputs.
            process.load_input_info()

            # Only load or create files in memory if it is a normal Process type. Multilooking always works from disk.
            if not isinstance(process, MultilookProcess):
                # Start with loading the inputs. If they are already loaded in memory then this step is not needed.
                process.load_input_data()
                # Then create the output memory files to write the output data
                process.create_memory()

            # Print processing
            print('Start processing ' + process.process_name + ' block ' + str(dat['block'] + 1) + ' out of ' +
                  str(dat['total_blocks']) + ' [' + str(dat['process_block_no']) + ' of total ' +
                  str(dat['total_process_block_no']) + '] for ' + process.out_processing_image.folder)
            # Then do the final calculations. (For multilooking apply the multilooking calculation)
            if isinstance(process, MultilookProcess):
                process.multilook_calculations()
            else:
                process.process_calculations()
        except:
            raise BrokenPipeError('Pipeline processing for ' + process.out_processing_image.folder + ' failed.')

        # Finally, if this step is saved to disk. Save the data from memory to disk.
        if save and not isinstance(process, MultilookProcess):
            process.save_to_disk()

    json_dicts = []
    json_files = []

    for processing_image in processing_images:
        json_dict, json_file = processing_image.get_json()
        json_dicts.append(json_dict)
        json_files.append(json_file)
    del processing_images, dat

    return [json_dicts, json_files]
