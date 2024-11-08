# This class analyzes one SLC using its orbits/datafile
# The main functionality of this class is:
#   - Read in a data and metafile in memory
#   - Write information in the shape of .json files
#   - Read information as .json files, including datafile and further processed files

# An image can either consist of one image and its corresponding information or a number of slices, which will be the
# bursts in the case of Sentinel. It is therefore important to say whether you want to work with the slices or the full
# image.

import os
import logging

from rippl.meta_data.image_processing_meta import ImageProcessingMeta
from rippl.meta_data.image_processing_concatenate import ImageConcatData
from rippl.meta_data.slc import SLC


class Interferogram(ImageConcatData):

    """
    :type data = ImageData
    :type slice_list = list
    """

    def __init__(self, folder, primary_slc='', secondary_slc='', reference_slc='', slice_list='', update_full_image=False):
        # Either give an xml_file or a res_file as input to define the meta_data of the image
        # Every image will contain slices (or bursts). This could be one or multiple.
        # Processing will be done based on these slices. Although some processes for which not everything needs to be
        # read in memory can be done seperately.

        if reference_slc == '':
            reference_slc = self.find_reference_slc(primary_slc, secondary_slc)
            if reference_slc == False:
                return
        if not os.path.exists(folder):
            succes = self.create_interferogram(folder, primary_slc, secondary_slc, reference_slc, slice_list)
            if not succes:
                logging.info('Not able to create new interferogram')
                return

        # Load the existing or created image data as an Image object.
        super().__init__(folder, slice_list, update_full_image)

        # Add the reference image. We use this image to reference other images too. If this image is missing coregistration
        # is not possible.
        if isinstance(reference_slc, SLC):
            self.reference_images['reference'] = reference_slc
        elif 'reference' in self.reference_paths.keys():
            self.reference_images['reference'] = SLC(os.path.dirname(self.reference_paths['reference']))
        if isinstance(primary_slc, SLC):
            self.reference_images['primary'] = primary_slc
        elif 'primary' in self.reference_paths.keys():
            self.reference_images['primary'] = SLC(os.path.dirname(self.reference_paths['primary']))
        if isinstance(secondary_slc, SLC):
            self.reference_images['secondary'] = secondary_slc
        elif 'secondary' in self.reference_paths.keys():
            self.reference_images['secondary'] = SLC(os.path.dirname(self.reference_paths['secondary']))

    @staticmethod
    def find_reference_slc(primary_slc, secondary_slc):
        # Check both primary and secondary for a reference image.
        if isinstance(primary_slc, SLC):
            if 'reference' in primary_slc.reference_images.keys():
                reference_slc = primary_slc.reference_images['reference'] = primary_slc
        elif isinstance(secondary_slc, SLC):
            if 'reference' in secondary_slc.reference_images.keys():
                reference_slc = secondary_slc.reference_images['reference'] = secondary_slc
        else:
            logging.info('No coregistration image found in secondary or primary image. Please define for interferogram image')
            return False

        return reference_slc

    @staticmethod
    def create_interferogram(folder, primary_slc, secondary_slc, reference_slc, slice_list=''):

        # Find coregistration image from primary or secondary if it exists
        if not isinstance(primary_slc, SLC) or not isinstance(secondary_slc, SLC) or not isinstance(reference_slc, SLC):
            logging.info('To create a new interferogram the primary, secondary and reference image should be an SLC object')
            return False
        if not os.path.exists(os.path.dirname(folder)):
            logging.info('Stack folder does not exist. Unable to create interferogram')
            return False

        slices = list(set(primary_slc.slice_names)
                      .intersection(set(secondary_slc.slice_names))
                      .intersection(set(reference_slc.slice_names)))

        if isinstance(slice_list, list):
            slices = list(set(slices).intersection(set(slice_list)))

        # Load all needed meta data.
        primary_slc.load_slice_meta()
        primary_slc.load_meta()
        secondary_slc.load_slice_meta()
        secondary_slc.load_meta()
        reference_slc.load_slice_meta()
        reference_slc.load_meta()

        # Create full image meta data
        Interferogram.create_interferogram_meta(folder, primary_slc, secondary_slc, reference_slc)

        # Create slices.
        for slice in slices:
            # Get path of new file and create folder
            slice_folder = os.path.join(folder, os.path.basename(folder) + '_' + slice)
            Interferogram.create_interferogram_meta(slice_folder, primary_slc.slice_data[slice], secondary_slc.slice_data[slice], reference_slc.slice_data[slice])

        return True

    @staticmethod
    def create_interferogram_meta(folder, primary_meta, secondary_meta, reference_meta):
        # Create metadata object and create folder
        meta = ImageProcessingMeta(folder)
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Get paths of primary, secondary, reference
        meta.reference_paths['primary'] = primary_meta.folder
        meta.reference_paths['secondary'] = secondary_meta.folder
        meta.reference_paths['reference'] = reference_meta.folder

        # Get orbit information
        primary_key = [key for key in primary_meta.orbits]
        meta.add_orbit(primary_meta.find_best_orbit('original'), 'primary')
        meta.add_orbit(secondary_meta.find_best_orbit('original'), 'secondary')
        meta.add_orbit(reference_meta.find_best_orbit('original'), 'reference')

        # Get readfiles information
        meta.add_readfile(primary_meta.readfiles['original'], 'primary')
        meta.add_readfile(secondary_meta.readfiles['original'], 'secondary')
        meta.add_readfile(reference_meta.readfiles['original'], 'reference')

        # Create header
        meta.create_header()

        # Write meta to .json
        meta.save_json()
