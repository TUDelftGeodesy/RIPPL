'''
This function evaluates a number of correlation windows over a (part) of an InSAR image. This can be used as a
method to do additional coregistration (in range/azimuth or both) or do particle tracking in part of an image,
in case of strong horizontal movement.
Main functions are:
    - Selection of windows over a certain region based on amplitudes and/or shapes. For masking we need a shape and
        lat/lon file over the same region. For amplitudes only the area.

'''

from rippl.meta_data.image_processing_data import ImageData


class CorrelationWindows(ImageData):

    def __init__(self, meta_slave, meta_master):
        # If the init function is called we will need a master and slave image.

        self.slave = ImageData(meta_slave, 'single')
        self.master = ImageData(meta_master, 'single')




    def find_windows(self, data, amplitude=True, n_blocks=100, size_az=50, size_ra=50, n_windows=1000):
        # If shape is given only the areas within this shape are selected.
        # This function is based on the master image only!
        # The procedure in this function is as follows:
        # - The total image is seperated in blocks of size * size
        # - All blocks that contains zeros are removed
        # - All blocks outside the shape are removed
        # - Then based on the value of amplitude:
        #       - True: The blocks highest average amplitude are selected (within every n blocks)
        #       - False: Every nth block is selected
        # - Window coordinates are calculated for the cross-correlation function


    def mask_images(self, shape, lat, lon, partly_outside=False):
        # This function is used to select the windows inside a certain region

    def cross_correlation(self, max_shift_az, max_shift_ra):
        # After the find_windows is run a cross-correlation for the windows in the master and slave is done

