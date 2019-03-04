'''
This function is used to create interferograms
'''

import os
from rippl.image import ImageData
from rippl.image import ImageData
from rippl.pipeline import Pipeline
import copy


class Interferogram(object):

    """
    :type slave: Image
    :type master: Image
    :type cmaster: Image
    :type folder: str
    :type slice_list: list
    """

    def __init__(self, folder='', slave='', master='', cmaster='', slice_list=''):
        # Either give an xml_file or a res_file as input to define the metadata of the image
        # Every image will contain slices (or bursts). This could be one or multiple.
        # Processing will be done based on these slices. Allthough some processes for which not everything needs to be
        # read in memory can be done seperately.

        # Read the image information
        self.folder = folder
        self.res_file = os.path.join(folder, 'info.res')

        # Create a new folder and .res file if it does not exist
        if not os.path.exists(self.res_file) and slave and master and cmaster:
            self.folder = os.path.join(os.path.dirname(slave.folder),
                                       os.path.basename(master.folder) + '_' + os.path.basename(slave.folder))
            print('New interferogram created at ' + self.folder)
            os.makedirs(self.folder)

            # Create main .res file
            self.res_file = os.path.join(self.folder, 'info.res')
            new_res = ImageData(self.res_file, 'interferogram')
            new_res.image_add_processing_step('coreg_readfiles', copy.copy(cmaster.res_data.processes['readfiles']))
            new_res.image_add_processing_step('coreg_orbits', copy.copy(cmaster.res_data.processes['orbits']))
            new_res.image_add_processing_step('coreg_crop', copy.copy(cmaster.res_data.processes['crop']))
            new_res.write(warn=False)
            new_res = []
        elif not os.path.exists(self.res_file):
            print('resfile of interferogram cannot be found and either slave/master/coreg master data is missing.')

        self.load_full_memmap()

        slice_folders = next(os.walk(self.folder))[1]
        self.slice_names = sorted([x for x in slice_folders if len(x) == 20])
        # If a slice_list is given, we limit to this list only.
        if slice_list:
            self.slice_names = list(set(self.slice_names) & set(slice_list))

        # Create .res files if one or more slices are missing
        if slave and master and cmaster:
            # Find set of master and slave slices
            if slice_list:
                slc_slices = (set(slave.slice_names) & set(master.slice_names) & set(cmaster.slice_names)) & set(slice_list)
            else:
                slc_slices = set(slave.slice_names) & set(master.slice_names) & set(cmaster.slice_names)
            new_slices = list(slc_slices - set(self.slice_names))
            self.slice_names = sorted(set(self.slice_names) | set(new_slices))

        # Add the master and slave image. We use this image to reference other images to. If these images are missing
        # making interferograms is not possible.
        self.master = master
        self.cmaster = cmaster
        self.slave = slave

        # Create the list of burst of this image and corresponding ifgs.
        self.slices = dict()
        self.slice_folders = [os.path.join(folder, x) for x in self.slice_names]
        self.slice_res_file = []

    def load_full_info(self):
        # Read the result file again.
        self.res_data = ImageData(self.res_file, res_type='interferogram')

    def load_full_memmap(self):
        # Read the result file again.
        self.res_data = ImageData(self.res_file, res_type='interferogram')
        self.res_data.read_data_memmap()

    def load_slice_info(self):
        # Load the information about the bursts (We do not load that information by default as it causes to much data in
        # memory for large stacks.

        if isinstance(self.cmaster, ImageData):
            self.cmaster.load_slice_info()

        for slice_folder, slice_name in zip(self.slice_folders, self.slice_names):
            res_path = os.path.join(slice_folder, 'info.res')
            self.slice_res_file.append(res_path)
            if os.path.exists(res_path):
                self.slices[slice_name] = ImageData(res_path, 'interferogram')
            else:
                self.slices[slice_name] = ImageData(res_path, 'interferogram')
                self.slices[slice_name].image_add_processing_step('coreg_readfiles', copy.copy(
                    self.cmaster.slices[slice_name].processes['readfiles']))
                self.slices[slice_name].image_add_processing_step('coreg_orbits', copy.copy(
                    self.cmaster.slices[slice_name].processes['orbits']))
                self.slices[slice_name].image_add_processing_step('coreg_crop', copy.copy(
                    self.cmaster.slices[slice_name].processes['crop']))
                self.slices[slice_name].geometry()
                self.slices[slice_name].res_path = res_path
                self.slices[slice_name].write(res_path)

    def load_slice_memmap(self):
        # Load the information about the bursts (We do not load that information by default as it causes to much data in
        # memory for large stacks.

        self.load_slice_info()
        for slice_name in self.slice_names:
            self.slices[slice_name].read_data_memmap()

    def rem_memmap(self):
        # Remove memmap information
        for slice_name in self.slice_names:
            self.slices[slice_name].clean_memmap_files()
        self.res_data.clean_memmap_files()

    def __call__(self, step, settings, coors, file_type='', slave='', cmaster='', master='', memory=500, cores=6,
                 parallel=True):
        # This calls the pipeline function for this step

        # Replace slave and master image if needed.
        if slave == '':
            slave = self.slave
        if master == '':
            master = self.master

        # Load all information from slices
        self.load_slice_info()
        self.rem_memmap()
        if isinstance(master, ImageData):
            master.load_slice_info()
            master.rem_memmap()
        if isinstance(slave, ImageData):
            slave.load_slice_info()
            slave.rem_memmap()
        if isinstance(cmaster, ImageData):
            cmaster.load_slice_info()
            cmaster.rem_memmap()

        # The main image is always seen as the slave image. Further ifg processing is not possible here.
        pipeline = Pipeline(memory=memory, cores=cores, slave=slave, master=master, cmaster=cmaster, ifg=self, parallel=parallel)
        pipeline(step, settings, coors, 'ifg', file_type=file_type)

        # Remove information from slices and load full image info again
        self.slices = dict()
        self.load_full_info()
        if isinstance(cmaster, ImageData):
            cmaster.load_slice_info()
            cmaster.load_full_memmap()
        if isinstance(master, ImageData):
            master.slices = dict()
            master.load_full_memmap()
        if isinstance(slave, ImageData):
            slave.slices = dict()
            slave.load_full_memmap()
