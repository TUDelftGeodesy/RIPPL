# This class analyzes one SLC using its orbits/datafile
# The main functionality of this class is:
#   - Read in a data and metafile in memory
#   - Write information in the shape of .json files
#   - Read information as .json files, including datafile and further processed files

# An image can either consist of one image and its corresponding information or a number of slices, which will be the
# bursts in the case of Sentinel. It is therefore important to say whether you want to work with the slices or the full
# image.

import os

from rippl.meta_data.image_processing_concatenate import ImageConcatData

class SLC(ImageConcatData):

    """
    :type reference_image = SLC
    :type data = ImageData
    :type slice_list = list
    """

    def __init__(self, folder, coreg_image='', slice_list='', update_full_image=False, adjust_date=False):
        # Either give an xml_file or a res_file as input to define the meta_data of the image
        # Every image will contain slices (or bursts). This could be one or multiple.
        # Processing will be done based on these slices. Although some processes for which not everything needs to be
        # read in memory can be done seperately.

        super(SLC, self).__init__(folder, slice_list, update_full_image, adjust_date=adjust_date)

        # Add the reference image. We use this image to reference other images too. If this image is missing coregistration
        # is not possible.
        if isinstance(coreg_image, SLC):
            self.reference_images['coreg'] = coreg_image
        elif 'coreg' in self.data.reference_paths.keys():
            self.reference_images['coreg'] = SLC(os.path.dirname(self.data.reference_paths['coreg']))

    def add_coreg_date(self):
        # Here we assume that the coreg image is already loaded.

        # Add the information of the coreg data as extra information.
        if 'coreg' in self.reference_images.keys():
            self.data.meta.reference_paths['coreg'] = self.reference_images['coreg'].meta.path
            self.data.meta.add_orbit(self.reference_images['coreg'].meta.orbits['original'], 'coreg')
            self.data.meta.add_readfile(self.reference_images['coreg'].meta.readfiles['original'], 'coreg')
