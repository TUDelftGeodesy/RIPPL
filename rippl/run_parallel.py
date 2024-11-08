# Function to run parallel package. For further details inspect the pipeline class.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.meta_data.image_processing_concatenate import ImageConcatData
import numpy as np
import datetime
import logging


def run_parallel(dat):
    """
    This function runs a parallel package for image processing.

    :param dict[list[Process] or list[bool]] dat:

    :return:
    """

    start_time = datetime.datetime.now()

    if 'concat_data' in dat.keys():
        process = dat['process']
        file_type = dat['file_type']
        coor = dat['coor']
        transition_type = dat['transition_type']
        remove_input = dat['remove_input']
        tmp_directory = dat['tmp_directory']
        output_type = dat['output_type']
        polarisation = dat['polarisation']
        data_id = dat['data_id']
        cut_off = dat['cut_off']
        overwrite = dat['overwrite']
        replace = dat['replace']

        image_data = dat['concat_data']             # type: ImageConcatData
        image_data.create_concatenate_image(process=process, file_type=file_type, coor=coor, tmp_directory=tmp_directory,
                                            transition_type=transition_type, remove_input=remove_input,
                                            output_type=output_type, polarisation=polarisation, data_id=data_id,
                                            cut_off=cut_off, overwrite=overwrite, replace=replace)
        image_data.remove_memmap_files()
        image_data.remove_slice_memmap()
        image_data.remove_memory_files()
        image_data.remove_slice_memory()
        del dat

        return True

    # Get all processing image datasets
    processing_images = []              # type: list[ImageProcessingData]
    image_keys = []

    # Load data as memmaps
    for process in dat['processes']:
        for key in process.processing_images.keys():
            if key not in image_keys:
                processing_images.append(process.processing_images[key])
                image_keys.append(key)

    # Loop over all processes.
    for process, save in zip(dat['processes'], dat['save_processes']):
        try:
            # Print processing
            dat['pixels'] = np.minimum(dat['pixels'], dat['total_pixels'] - dat['s_pix'])
            dat['lines'] = np.minimum(dat['lines'], dat['total_lines'] - dat['s_lin'])
            logging.info('Start processing ' + process.process_name + ' chunk ' + str(dat['chunk']) + ' out of ' +
                  str(dat['total_chunks']) + ' [' + str(dat['process_chunk_no']) + ' of total ' +
                  str(dat['total_process_chunk_no']) + ' in this processing block] for ' + process.out_processing_image.folder)
            logging.info('Processing image region from lines ' + str(dat['s_lin'] + 1) + ' > ' + str(dat['s_lin'] + dat['lines'])
                  + ' and pixels ' + str(dat['s_pix'] + 1) + ' > ' + str(dat['s_pix'] + dat['pixels']) +
                  ' with size ' + str(dat['lines']) + ' x ' + str(dat['pixels']) +
                  ' from total image size ' + str(dat['total_lines']) + ' x ' + str(dat['total_pixels']))
            logging.info('Processing start time is ' + str(datetime.datetime.now()))

            # First init the process inputs and outputs.
            process.load_input_info()
            # Load the output files if needed
            if save:
                process.load_output_data_files()

            # Start with loading the inputs. If they are already loaded in memory then this step is not needed.
            process.load_input_data(scratch_disk_dir=dat['scratch_disk_dir'], internal_memory_dir=dat['internal_memory_dir'])
            # Then create the output memory files to write the output data
            process.create_memory()

            # Then do the final calculations. (For multilooking apply the multilooking calculation)
            process.process_calculations()

        except Exception as e:
            raise BrokenPipeError('Pipeline processing for ' + process.out_processing_image.folder + ' failed. ' + str(e))

        # Finally, if this step is saved to disk. Save the data from memory to disk.
        if save:
            process.save_to_disk()

    logging.info('Finished processing pipeline in ' + str(datetime.datetime.now() - start_time))
    json_dicts = []
    json_files = []

    for processing_image in processing_images:
        json_dict, json_file = processing_image.get_json()
        json_dicts.append(json_dict)
        json_files.append(json_file)
        processing_image.remove_memory_files()
        processing_image.remove_memmap_files()
    del processing_images, dat

    return [json_dicts, json_files]
