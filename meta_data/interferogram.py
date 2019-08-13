# This class analyzes one SLC using its orbits/datafile
# The main functionality of this class is:
#   - Read in a data and metafile in memory
#   - Write information in the shape of .json files
#   - Read information as .json files, including datafile and further processed files

# An image can either consist of one image and its corresponding information or a number of slices, which will be the
# bursts in the case of Sentinel. It is therefore important to say whether you want to work with the slices or the full
# image.

import os

from rippl.meta_data.image_meta import ImageMeta
from rippl.meta_data.image import Image
from rippl.meta_data.slc import SLC


class Interferogram(Image):

    """
    :type reference_image = SLC
    :type data = ImageData
    :type slice_list = list
    """

    def __init__(self, folder, master_image='', slave_image='', slice_list='', update_full_image=False):
        # Either give an xml_file or a res_file as input to define the meta_data of the image
        # Every image will contain slices (or bursts). This could be one or multiple.
        # Processing will be done based on these slices. Although some processes for which not everything needs to be
        # read in memory can be done seperately.

        if not os.path.exists(folder):
            succes = self.create_interferogram(folder, master_image, slave_image, slice_list)
            if not succes:
                print('Not able to create new interferogram')
                return

        # Load the existing or created image data as an Image object.
        super.__init__(folder, slice_list, update_full_image)

        # Add the reference image. We use this image to reference other images too. If this image is missing coregistration
        # is not possible.
        coreg_image = self.find_coreg_image(master_image, slave_image)
        if isinstance(coreg_image, SLC):
            self.reference_images['coreg'] = coreg_image
        elif 'coreg' in self.data.reference_paths.keys():
            self.reference_images['coreg'] = SLC(os.path.dirname(self.data.reference_paths['coreg']))
        if isinstance(master_image, SLC):
            self.reference_images['master'] = master_image
        elif 'master' in self.data.reference_paths.keys():
            self.reference_images['master'] = SLC(os.path.dirname(self.data.reference_paths['master']))
        if isinstance(slave_image, SLC):
            self.reference_images['slave'] = slave_image
        elif 'slave' in self.data.reference_paths.keys():
            self.reference_images['slave'] = SLC(os.path.dirname(self.data.reference_paths['slave']))

    @staticmethod
    def find_coreg_image(master_image, slave_image):
        # Check both master and slave for a coregistration image.
        if isinstance(master_image, SLC):
            if 'coreg' in master_image.reference_images.keys():
                coreg_image = master_image.reference_images['coreg'] = master_image
        if isinstance(slave_image, SLC):
            if 'coreg' in slave_image.reference_images.keys():
                coreg_image = slave_image.reference_images['coreg'] = slave_image

        return coreg_image

    @staticmethod
    def create_interferogram(folder, master_image, slave_image, slice_list=''):

        coreg_image = Interferogram.find_coreg_image(master_image, slave_image)

        # Find coregistration image from master or slave if it exists
        if not isinstance(master_image, SLC) or not isinstance(slave_image, SLC) or not isinstance(coreg_image, SLC):
            print('To create a new interferogram the master, slave and coreg image should be an SLC object')
            return False
        if not os.path.exists(os.path.dirname(folder)):
            print('Stack folder does not exist. Unable to create interferogram')
            return False

        slices = list(set(master_image.slice_names)
                      .intersection(set(slave_image.slice_names))
                      .intersection(set(coreg_image.slice_names)))

        if isinstance(slice_list, list):
            slices = list(set(slices).intersection(set(slice_list)))

        # Load all needed meta data.
        master_image.load_full_meta()
        master_image.load_slice_meta()
        slave_image.load_full_meta()
        slave_image.load_slice_meta()
        coreg_image.load_full_meta()
        coreg_image.load_slice_meta()

        # Create full image meta data
        Interferogram.create_interferogram_meta(folder, master_image.data, slave_image.data, coreg_image.data)

        # Create slices.
        for slice in slices:
            # Get path of new file and create folder
            slice_folder = os.path.join(folder, slice)
            Interferogram.create_interferogram_meta(slice_folder, master_image.slice_data[slice],
                                                    slave_image.slice_data[slice]. coreg_image.slice_data[slice])

    @staticmethod
    def create_interferogram_meta(folder, master_meta, slave_meta, coreg_meta):
        # Create metadata object and create folder
        meta = ImageMeta(os.path.join(folder, 'info.json'))
        os.mkdir(os.path.join(folder))

        # Get paths of master, slave, coreg
        meta.reference_paths['master'] = master_meta.path
        meta.reference_paths['slave'] = slave_meta.path
        meta.reference_paths['reference'] = coreg_meta.path

        # Get orbit information
        meta.add_orbit(master_meta.orbits['original'], 'master')
        meta.add_orbit(slave_meta.orbits['original'], 'slave')
        meta.add_orbit(coreg_meta.orbits['original'], 'coreg')

        # Get readfiles information
        meta.add_orbit(master_meta.orbits['original'], 'master')
        meta.add_orbit(slave_meta.orbits['original'], 'slave')
        meta.add_orbit(coreg_meta.orbits['original'], 'coreg')

        # Write meta to .json
        meta.update_json()
