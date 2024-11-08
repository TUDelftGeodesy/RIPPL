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

    def __init__(self, folder, reference_image='', slice_list='', update_full_image=False):
        # Either give an xml_file or a res_file as input to define the meta_data of the image
        # Every image will contain slices (or bursts). This could be one or multiple.
        # Processing will be done based on these slices. Although some processes for which not everything needs to be
        # read in memory can be done seperately.

        super().__init__(folder, slice_list, update_full_image)

        # Add the reference image. We use this image to reference other images too. If this image is missing coregistration
        # is not possible.
        if isinstance(reference_image, SLC):
            self.reference_images['reference'] = reference_image
        elif 'reference' in self.reference_paths.keys():
            self.reference_images['reference'] = SLC(os.path.dirname(self.reference_paths['reference']))

    def add_reference_date(self):
        # Here we assume that the reference image is already loaded.

        # Add the information of the reference data as extra information.
        if 'reference' in self.reference_images.keys():
            self.reference_paths['reference'] = self.reference_images['coreg'].path
            self.add_orbit(self.reference_images['coreg'].orbits['original'], 'coreg')
            self.add_readfile(self.reference_images['coreg'].readfiles['original'], 'coreg')
