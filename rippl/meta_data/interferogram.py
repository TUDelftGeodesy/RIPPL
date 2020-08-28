# This class analyzes one SLC using its orbits/datafile
# The main functionality of this class is:
#   - Read in a data and metafile in memory
#   - Write information in the shape of .json files
#   - Read information as .json files, including datafile and further processed files

# An image can either consist of one image and its corresponding information or a number of slices, which will be the
# bursts in the case of Sentinel. It is therefore important to say whether you want to work with the slices or the full
# image.

import os

from rippl.meta_data.image_processing_meta import ImageProcessingMeta
from rippl.meta_data.image_processing_concatenate import ImageConcatData
from rippl.meta_data.slc import SLC


class Interferogram(ImageConcatData):

    """
    :type reference_image = SLC
    :type data = ImageData
    :type slice_list = list
    """

    def __init__(self, folder, master_slc='', slave_slc='', coreg_slc='', slice_list='', update_full_image=False, adjust_date=False):
        # Either give an xml_file or a res_file as input to define the meta_data of the image
        # Every image will contain slices (or bursts). This could be one or multiple.
        # Processing will be done based on these slices. Although some processes for which not everything needs to be
        # read in memory can be done seperately.

        if coreg_slc == '':
            coreg_slc = self.find_coreg_slc(master_slc, slave_slc)
            if coreg_slc == False:
                return
        if not os.path.exists(folder):
            succes = self.create_interferogram(folder, master_slc, slave_slc, coreg_slc, slice_list)
            if not succes:
                print('Not able to create new interferogram')
                return

        # Load the existing or created image data as an Image object.
        super(Interferogram, self).__init__(folder, slice_list, update_full_image, adjust_date=adjust_date)

        # Add the reference image. We use this image to reference other images too. If this image is missing coregistration
        # is not possible.
        if isinstance(coreg_slc, SLC):
            self.reference_images['coreg'] = coreg_slc
        elif 'coreg' in self.data.reference_paths.keys():
            self.reference_images['coreg'] = SLC(os.path.dirname(self.data.reference_paths['coreg']))
        if isinstance(master_slc, SLC):
            self.reference_images['master'] = master_slc
        elif 'master' in self.data.reference_paths.keys():
            self.reference_images['master'] = SLC(os.path.dirname(self.data.reference_paths['master']))
        if isinstance(slave_slc, SLC):
            self.reference_images['slave'] = slave_slc
        elif 'slave' in self.data.reference_paths.keys():
            self.reference_images['slave'] = SLC(os.path.dirname(self.data.reference_paths['slave']))

    @staticmethod
    def find_coreg_slc(master_slc, slave_slc):
        # Check both master and slave for a coregistration image.
        if isinstance(master_slc, SLC):
            if 'coreg' in master_slc.reference_images.keys():
                coreg_slc = master_slc.reference_images['coreg'] = master_slc
        elif isinstance(slave_slc, SLC):
            if 'coreg' in slave_slc.reference_images.keys():
                coreg_slc = slave_slc.reference_images['coreg'] = slave_slc
        else:
            print('No coregistration image found in slave or master image. Please define for interferogram image')
            return False

        return coreg_slc

    @staticmethod
    def create_interferogram(folder, master_slc, slave_slc, coreg_slc, slice_list=''):

        # Find coregistration image from master or slave if it exists
        if not isinstance(master_slc, SLC) or not isinstance(slave_slc, SLC) or not isinstance(coreg_slc, SLC):
            print('To create a new interferogram the master, slave and coreg image should be an SLC object')
            return False
        if not os.path.exists(os.path.dirname(folder)):
            print('Stack folder does not exist. Unable to create interferogram')
            return False

        slices = list(set(master_slc.slice_names)
                      .intersection(set(slave_slc.slice_names))
                      .intersection(set(coreg_slc.slice_names)))

        if isinstance(slice_list, list):
            slices = list(set(slices).intersection(set(slice_list)))

        # Load all needed meta data.
        master_slc.load_full_meta()
        master_slc.load_slice_meta()
        slave_slc.load_full_meta()
        slave_slc.load_slice_meta()
        coreg_slc.load_full_meta()
        coreg_slc.load_slice_meta()

        # Create full image meta data
        Interferogram.create_interferogram_meta(folder, master_slc.data, slave_slc.data, coreg_slc.data)

        # Create slices.
        for slice in slices:
            # Get path of new file and create folder
            slice_folder = os.path.join(folder, slice)
            Interferogram.create_interferogram_meta(slice_folder, master_slc.slice_data[slice], slave_slc.slice_data[slice], coreg_slc.slice_data[slice])

        return True

    @staticmethod
    def create_interferogram_meta(folder, master_meta, slave_meta, coreg_meta):
        # Create metadata object and create folder
        meta = ImageProcessingMeta(folder)
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Get paths of master, slave, coreg
        meta.reference_paths['master'] = master_meta.folder
        meta.reference_paths['slave'] = slave_meta.folder
        meta.reference_paths['reference'] = coreg_meta.folder

        # Get orbit information
        master_key = [key for key in master_meta.orbits]
        meta.add_orbit(master_meta.find_best_orbit('original'), 'master')
        meta.add_orbit(slave_meta.find_best_orbit('original'), 'slave')
        meta.add_orbit(coreg_meta.find_best_orbit('original'), 'coreg')

        # Get readfiles information
        meta.add_readfile(master_meta.readfiles['original'], 'master')
        meta.add_readfile(slave_meta.readfiles['original'], 'slave')
        meta.add_readfile(coreg_meta.readfiles['original'], 'coreg')

        # Create header
        meta.create_header()

        # Write meta to .json
        meta.save_json()
