# Function to run parallel package. For further details inspect the pipeline class.
from rippl.meta_data.process import Process


def run_parallel(dat):
    """
    This function runs a parallel package for image processing.

    :param dict[list[Process] or list[bool]] dat:

    :return:
    """

    # Loop over all processes.
    for process, save in zip(dat['processes'], dat['save_processes']):

        # Start with loading the inputs. If they are already loaded in memory then this step is not needed.
        process.load_input_data()
        # Then create the output memory files to write the output data
        process.create_memory()
        # Then do the final calculation
        process.process_calculations()

        # Finally, if this step is saved to disk. Save the data from memory to disk.
        if save:
            process.save_to_disk()
