'''
This function is used to create interferograms
'''

import os
from image_metadata import ImageMetadata
from image_data import ImageData
from image import Image
from processing_steps.unwrap import Unwrap
from pipeline import Pipeline


class Interferogram(object):

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

        self.res_data = ImageData(self.res_file, res_type='interferogram')

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
            self.slice_names = sorted(set(self.slice_names) | set(new_slices))

        # Add the master and slave image. We use this image to reference other images to. If these images are missing
        # making interferograms is not possible.
        self.master = master
        self.slave = slave

        # Create the list of burst of this image
        self.slices = dict()
        slice_folders = [os.path.join(folder, x) for x in self.slice_names]

    def __call__(self, step, settings, coors, file_type='', slice=True, slave='', cmaster='', master='', memory=500, cores=6,
                 parallel=True):
        # This calls the pipeline function for this step

        # Replace slave and master image if needed.
        if slave == '':
            slave = self.slave
        if master == '':
            slave = self.master

        # The main image is always seen as the slave image. Further ifg processing is not possible here.
        pipeline = Pipeline(memory=memory, cores=cores, slave=slave, master=master, cmaster=cmaster, ifg=self, parallel=parallel)
        pipeline(step, settings, coors, 'ifg', slice=slice, file_type=file_type)

    def unwrap(self, multilook='', offset=''):
        # Applies the unwrapping for the full interferogram.
        # Load the master and slave image

        unwrap = Unwrap(self, step='interferogram', offset=offset, multilook=multilook)
        unwrap()
