# Function to run parallel package. For further details inspect the pipeline class.
import gc

from rippl.meta_data.process import Process
from rippl.meta_data.multilook_process import MultilookProcess
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.meta_data.image_processing_concatenate import ImageConcatData
import numpy as np
import datetime


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
        image_data.remove_full_memmap()
        image_data.remove_slice_memmap()
        image_data.remove_full_memory()
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
    for process, save, memory_in in zip(dat['processes'], dat['save_processes'], dat['memory_in']):
        try:
            # Load needed input shapes
            process.load_coordinate_system_sizes(find_out_coor=False)
            # First init the process inputs and outputs.
            process.load_input_info()
            # Load the output files if needed
            if save:
                process.load_output_data_files()

            # Only load or create files in memory if it is a normal Process type. Multilooking always works from disk.
            if not isinstance(process, MultilookProcess):
                # Start with loading the inputs. If they are already loaded in memory then this step is not needed.
                process.load_input_data(tmp_directory=dat['tmp_directory'], coreg_tmp_directory=dat['coreg_tmp_directory'])
                # Then create the output memory files to write the output data
                process.create_memory()
            else:
                # Load the input memory mapped files
                process.load_input_data_files(tmp_directory=dat['tmp_directory'], coreg_tmp_directory=dat['coreg_tmp_directory'])

            # Print processing
            dat['pixels'] = np.minimum(dat['pixels'], dat['total_pixels'] - dat['s_pix'])
            dat['lines'] = np.minimum(dat['lines'], dat['total_lines'] - dat['s_lin'])
            print('Start processing ' + process.process_name + ' block ' + str(dat['block'] + 1) + ' out of ' +
                  str(dat['total_blocks']) + ' [' + str(dat['process_block_no']) + ' of total ' +
                  str(dat['total_process_block_no']) + '] for ' + process.out_processing_image.folder)
            print('Processing image region from lines ' + str(dat['s_lin'] + 1) + ' > ' + str(dat['s_lin'] + dat['lines'])
                  + ' and pixels ' + str(dat['s_pix'] + 1) + ' > ' + str(dat['s_pix'] + dat['pixels']) +
                  ' with size ' + str(dat['lines']) + ' x ' + str(dat['pixels']) +
                  ' from total image size ' + str(dat['total_lines']) + ' x ' + str(dat['total_pixels']))
            print('Processing start time is ' + str(datetime.datetime.now()))

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

    print('Finished processing pipeline in ' + str(datetime.datetime.now() - start_time))
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
