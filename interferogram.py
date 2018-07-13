'''
This function is used to creat interferograms

'''

import os
from doris_processing.image_metadata import ImageMetadata
from doris_processing.image_data import ImageData
from doris_processing.image import Image
from doris_processing.processing_steps.unwrap import Unwrap


class Interferogram(ImageData):

    """
    :type slave: Image
    :type master: Image
    :type folder: str
    :type slice_list: list

    """

    def __init__(self, folder='', slave='', master='', slice_list=''):
        # Either give an xml_file or a res_file as input to define the metadata of the image
        # Every image will contain slices (or bursts). This could be one or multiple.
        # Processing will be done based on these slices. Allthough some processes for which not everything needs to be
        # read in memory can be done seperately.

        # Read the image information
        self.folder = folder
        self.res_file = os.path.join(folder, 'info.res')

        # Create a new folder and .res file if it does not exist
        if not os.path.exists(self.res_file) and slave and master:
            self.folder = os.path.join(os.path.dirname(slave.folder),
                                       os.path.basename(slave.folder) + '_' + os.path.basename(master.folder))
            print('New interferogram created at ' + self.folder)
            os.makedirs(self.folder)

            # Create main .res file
            self.res_file = os.path.join(self.folder, 'info.res')
            new_res = ImageMetadata('', 'interferogram')
            new_res.write(self.res_file)
            del new_res

        ImageData.__init__(self, self.res_file, res_type='interferogram')

        slice_folders = next(os.walk(folder))[1]
        self.slice_names = sorted([x for x in slice_folders if len(x) == 20])
        # If a slice_list is given, we limit to this list only.
        if slice_list:
            self.slice_names = list(set(self.slice_names) & set(slice_list))

        # Create .res files if one or more slices are missing
        if slave and master:
            # Find set of master and slave slices
            if slice_list:
                slc_slices = (set(slave.slice_names) & set(master.slice_names)) & set(slice_list)
            else:
                slc_slices = set(slave.slice_names) & set(master.slice_names)
            new_slices = list(slc_slices - set(self.slice_names))

            #new_res = ImageMetadata('', 'interferogram')
            #for new_slice in new_slices:
            #    self.res_file = os.path.join(self.folder, new_slice, 'info.res')
            #    new_res.write(self.res_file)

            #del new_res
            # update slice list
            self.slice_names = sorted(set(self.slice_names) | set(new_slices))

        # Add the master and slave image. We use this image to reference other images to. If these images are missing
        # making interferograms is not possible.
        self.master = master
        self.slave = slave

        # Create the list of burst of this image
        self.slices = dict()
        slice_folders = [os.path.join(folder, x) for x in self.slice_names]

        # Read the individual slices ('slice_' + str(number) + '_swath_' + str(swath_no) + '_' + pol)
        # Length of these folders should be 20 characters
        #for slice_folder, slice_name in zip(slice_folders, self.slice_names):
        #    self.slices[os.path.basename(slice_folder)] = ImageData(os.path.join(slice_folder, 'info.res'), 'interferogram')

    def unwrap(self, multilook='', offset=''):
        # Applies the unwrapping for the full interferogram.
        # Load the master and slave image

        unwrap = Unwrap(self, step='interferogram', offset=offset, multilook=multilook)
        unwrap()
