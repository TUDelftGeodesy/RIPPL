import numpy as np
import copy

from rippl.meta_data.image_processing_concatenate import ImageConcatData
from rippl.meta_data.image_processing_data import ImageProcessingMeta, ImageProcessingData
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.process_data import ProcessMeta, ProcessData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.coor_concatenate import CoorConcatenate

class Concatenate():

    """
    This class combines functionalities to concatenate a number of slices to a full image.
    """

    def __init__(self, concat_image, process, file_type, coor, data_id='', polarisation='', overwrite=False, output_type='disk'):
        """
        Load the dataset which should be concatenated and load the input parameters.

        :param ImageConcatData concat_image: Image where the concatenation should be done
        :param str process: Process type we want to concatenate
        :param str file_type: Process type that we want to concatenate
        :param CoordinateSystem coor: Coordinatesystem of input/output images
        :param str data_id: Data ID. Generally empty
        :param str polarisation: Polarisation of dataset
        :param bool overwrite: Do we want to overwrite if data already exists
        :param str output_type: Should concatenation be done on disk or memory
        """

        if not isinstance(concat_image, ImageConcatData):
            raise TypeError('Variable concat_image should be an ImageConcatData')

        self.concat_image = concat_image
        self.process = process
        self.file_type = file_type
        self.coordinates = coor
        self.data_id = data_id
        self.polarisation = polarisation

        # Images to concatenate
        self.concat_data = []               # type: ImageData
        self.slice_data = []                # type: list(ImageData)
        self.slice_coordinates = []         # type: list(CoordinateSystem)
        self.slice_names = []

        # Do we overwrite already existing images?
        self.overwrite = overwrite

        # Type of output (disk or memory)
        self.output_type = output_type

        # Init the lines/pixels weights.
        self.line_weights = dict()             # type: dict(np.ndarray)
        self.pixel_weights = dict()            # type: dict(np.ndarray)

    def load_data(self):
        """
        Load data and check whether they exist / have the right format etc.

        :return:
        """

        slice_names, processes, process_ids, coordinates, in_coordinates, file_types, images_out = \
            self.concat_image.concat_image_data_iterator(processes=[self.process],
                                                        coordinates=[self.coordinates],
                                                        file_types=[self.file_type],
                                                        data_ids=[self.data_id],
                                                        polarisations=[self.polarisation],
                                                        slices=True,
                                                        full_image=False,
                                                        load_memmap=False)

        if len(process_ids) == 0:
            self.coordinates.create_coor_id()
            raise FileNotFoundError('No slices found for concatenation for process ' + self.process +
                                    ' ,file type ' + self.file_type +
                                    ' ,data id ' + self.data_id +
                                    ' ,polarisation ' + self.polarisation +
                                    ' and coordinates ' + self.coordinates.short_id_str)

        data_ids = []
        polarisations = []
        in_coordinates_str = []
        for process_id in process_ids:
            proc, coor_str, in_coor_str, id_str, pol_str = ProcessMeta.split_process_id(process_id)
            data_ids.append(id_str)
            polarisations.append(pol_str)
            in_coordinates_str.append(in_coor_str)

        sort_ids = np.argsort(slice_names)
        self.slice_data = list(np.array(images_out)[sort_ids])
        self.slice_coordinates = list(np.array(coordinates)[sort_ids])
        self.slice_names = list(np.array(slice_names)[sort_ids])
        self.process_id = process_ids[0]
        self.data_id = data_ids[0]
        self.polarisation = polarisations[0]
        self.in_coordinates = in_coordinates[0]

        if len(slice_names) == 0:
            print('No slices found to concatenate!')
            return
        if len(coordinates) != len(self.concat_image.slice_names):
            print('Number of slices and found input data images is not the same. Aborting concatenation')
            return
        if not len(set(data_ids)) == 1 or not len(set(polarisations)) == 1 or not len(set(in_coordinates_str)) == 1:
            print('Please specify polarisation, data id or in coordinates to select the right process')
            return

    def create_image(self, tmp_directory=''):
        """
        Create the output image.

        :return:
        """

        if len(self.coordinates.shape) == 2:
            print('Using predefined grid size for concatenation of slices')
            concat = CoorConcatenate(self.slice_coordinates, concat_coor=self.coordinates, adjust_date=self.concat_image.adjust_date)
        else:
            concat = CoorConcatenate(self.slice_coordinates, adjust_date=self.concat_image.adjust_date)
        self.slice_coordinates = concat.sync_coors
        concat_coor = concat.concat_coor

        # Based on the new coordinate system create a new process.
        # Check if process already exists in
        process_id_full = self.concat_image.concat_image_data_iterator(processes=[self.process],
                                                                      coordinates=[concat_coor],
                                                                      file_types=[],
                                                                      data_ids=[self.data_id],
                                                                      polarisations=[self.polarisation],
                                                                      slices=False,
                                                                      full_image=True)[2]
        # If the process exists.
        if len(process_id_full) > 0:
            process = self.concat_image.data.processes_data[self.process][process_id_full[0]]

            # Check if the image already exists.
            image = process.process_data_iterator(file_types=[self.file_type])[-1]
            # Check if the file exists on disk or in memory.
            if len(image) > 0:
                if (image[0].check_disk_file() or image[0].check_memory_file()) and self.overwrite == False:
                    print('Concatenated dataset already exists. If you want to overwrite set overwrite to True')
                    return False
        else:
            process = ProcessData(folder=self.concat_image.folder,
                                  process_name=self.process,
                                  coordinates=concat_coor, 
                                  data_id=self.data_id, 
                                  polarisation=self.polarisation, 
                                  in_coordinates=self.in_coordinates)
            self.concat_image.data.add_process(process)

        # Create an empty output file.
        process.add_process_image(self.file_type, self.slice_data[0].dtype)
        self.concat_data = process.process_data_iterator([self.file_type])[-1][0]          # type: ImageData
        
        if self.output_type == 'memory':
            self.concat_data.new_memory_data(self.concat_data.coordinates.shape)
        else:
            if tmp_directory:
                self.tmp_directory = tmp_directory
                self.concat_data.create_disk_data(tmp_directory=tmp_directory)
            else:
                self.concat_data.create_disk_data(overwrite=True)

        return True

    def concatenate(self, transition_type='coverage_cut_off', replace=False, cut_off=10, remove_input=False):
        """
        Here the actual concatenation is done.

        :param str transition_type: This defines the type of transition between the
        :return:
        """

        if transition_type != 'coverage_cut_off':
            self.transition_zones(transition_type, cut_off)

        for image, coordinates, slice_name in zip(self.slice_data, self.slice_coordinates, self.slice_names):

            self.calc_coverage(transition_type=transition_type, cut_off=cut_off, coordinates=coordinates, slice_name=slice_name)

            lin_0 = int(coordinates.first_line - self.concat_data.coordinates.first_line)
            pix_0 = int(coordinates.first_pixel - self.concat_data.coordinates.first_pixel)
            lin_size = coordinates.shape[0]
            pix_size = coordinates.shape[1]

            if tuple(image.memory['meta']['shape']) == tuple(image.shape):
                if replace:
                    # Set data that is going to be filled to zero.
                    self.concat_data.memory['data'][lin_0:lin_0 + lin_size, pix_0:pix_0 + pix_size] = 0

                if self.output_type == 'memory':
                    self.concat_data.memory['data'][lin_0:lin_0 + lin_size, pix_0:pix_0 + pix_size] += \
                        image.memory['data'] * self.coverage
                else:
                    self.concat_data.memory['data'][lin_0:lin_0 + lin_size, pix_0:pix_0 + pix_size] += \
                        image.memory2disk(image.memory['data'] * self.coverage, image.dtype)
            elif image.load_disk_data():
                if replace:
                    # Set data that is going to be filled to zero.
                    self.concat_data.disk['data'][lin_0:lin_0 + lin_size, pix_0:pix_0 + pix_size] = 0

                if self.output_type == 'memory':
                    self.concat_data.memory['data'][lin_0:lin_0 + lin_size, pix_0:pix_0 + pix_size] += \
                        image.disk2memory(image.disk['data'], image.dtype)  * self.coverage
                else:
                    if image.dtype in ['complex_short', 'complex_int']:
                        self.concat_data.disk['data'][lin_0:lin_0 + lin_size, pix_0:pix_0 + pix_size] = \
                            image.memory2disk((image.disk2memory(image.disk['data'], image.dtype) * self.coverage).astype(np.complex64) +
                                              image.disk2memory(copy.copy(self.concat_data.disk['data'][lin_0:lin_0 + lin_size, pix_0:pix_0 + pix_size]), image.dtype).astype(np.complex64), image.dtype)
                    else:
                        self.concat_data.disk['data'][lin_0:lin_0 + lin_size, pix_0:pix_0 + pix_size] += \
                            image.disk['data'] * self.coverage

                image.remove_disk_data_memmap()  # type: ImageData
            else:
                raise TypeError('Not possible to load either memory or disk data')

            if remove_input:
                image.remove_disk_data()
                image.remove_memory_data()

        # Copy from temporary file if needed.
        if self.concat_data.tmp_path:           # type: ImageData
            self.concat_data.save_tmp_data()

        # Finally save the .json data.
        self.concat_image.data.save_json()
        print('Finished concatenation ' + self.concat_data.process_name + ' of ' + self.concat_data.folder)

    def full_weight(self):
        """
        This function will simply use all the data that is available from the different images. This is suitable when
        you only look at interferogram

        :return:
        """

        for coordinates, slice_name in zip(self.slice_coordinates, self.slice_names):
            # We give weights of one to all lines/pixels
            self.line_weights[slice_name] = np.ones(coordinates.shape[0])
            self.pixel_weights[slice_name] = np.ones(coordinates.shape[1])

    def calc_coverage(self, transition_type='coverage_cut_off', cut_off=10, coordinates='', slice_name=''):
        """
        Here we calculate the coverage the current burst, including cutoffs

        """

        if not isinstance(coordinates, CoordinateSystem):
            return

        if transition_type == 'full_weight':
            self.coverage = np.ones(coordinates.shape)
            return
        elif transition_type != 'coverage_cut_off':
            self.coverage = self.line_weights[slice_name][:, None] * self.pixel_weights[None, :][slice_name]
            return

        # Otherwise we will have individual coverage cases.
        self.coverage = np.zeros(coordinates.shape)
        self.coverage[cut_off:-cut_off, cut_off:-cut_off] = 1
        shape = coordinates.shape

        # Then loop over the available other images, and if they are earlier in the list, remove that part of the coverage
        for name, coor in zip(self.slice_names, self.slice_coordinates):

            # Only the images upto the current slice are used. So slices with smaller burst of swath numbers are preferred.
            if name == slice_name:
                break

            first_line = coor.first_line - coordinates.first_line + cut_off
            first_pixel = coor.first_pixel - coordinates.first_pixel + cut_off

            last_line = first_line + coor.shape[0] - 2 * cut_off
            last_pixel = first_pixel + coor.shape[1] - 2 * cut_off

            # Check if there is an overlap.
            if (first_line < shape[0] and last_line >= 0) and (first_pixel < shape[1] and last_pixel >= 0):
                first_line = np.maximum(first_line, 0)
                last_line = np.minimum(last_line, shape[0])
                first_pixel = np.maximum(first_pixel, 0)
                last_pixel = np.minimum(last_pixel, shape[1])

                self.coverage[first_line:last_line, first_pixel:last_pixel] = 0
            else:
                continue

    def transition_zones(self, transition_type='linear', cut_off=10):
        """
        This function creates transition zones from one burst to the other.

        :param str transition_type: Type of transition zone between
        :param cut_off: The cut off value to prevent using zeros at the boundaries. Note that this will be corrected
                        using the multilooking/oversampling factor
        :return:
        """

        if self.coordinates.grid_type != 'radar_coordinates':
            raise TypeError('Concatenating using transition zones can only be done with radar coordinates and not with '
                            'geographic or projected grids.')
        cut_off = [int(np.ceil(cut_off / (self.coordinates.multilook[0] / self.coordinates.oversample[0]))),
                   int(np.ceil(cut_off / (self.coordinates.multilook[1] / self.coordinates.oversample[1])))]

        readfile_names = [list(self.concat_image.slice_data[slice_name].readfiles.keys())[0] for slice_name in self.slice_names]
        swaths = np.array([self.concat_image.slice_data[slice_name].readfiles[readfile_name].swath for
                           slice_name, readfile_name in zip(self.slice_names, readfile_names)])
        slice_nums = np.array([int(slice_name[6:9]) for slice_name in self.slice_names])

        for swath in sorted(list(set(swaths))):

            # Check the first and last pixel of the full swath.
            swath_slice_ids = np.ravel(np.argwhere(swaths == swath))
            swath_slice_coordinates = [self.slice_coordinates[id] for id in swath_slice_ids]
            swath_slice_names = [self.slice_names[id] for id in swath_slice_ids]
            swath_first_pixel = np.max([coor.first_pixel for coor in swath_slice_coordinates])
            swath_last_pixel = np.min([coor.first_pixel + coor.shape[1] for coor in swath_slice_coordinates])
            swath_pixel_weights = np.ones(swath_last_pixel - swath_first_pixel)

            # Check transition of one swath to the other. To prevent problems with multiple burst overlapping the
            # transition zone of the full swath is the same.
            if swath - 1 in swaths:
                before_slice_coordinates = [self.slice_coordinates[id] for id in np.ravel(np.argwhere(swaths == swath - 1))]
                before_last_pixel = np.min([coor.first_pixel + coor.shape[1] for coor in before_slice_coordinates])

                if (before_last_pixel - swath_first_pixel) < 1:
                    raise ValueError('Not possible to concatenate using this transition type. Use coverage_cut_off instead.')
                overlap = np.zeros(before_last_pixel - swath_first_pixel)

                if transition_type == 'linear':
                    transition_length = len(overlap) - cut_off[1] * 2
                    overlap[cut_off[1]:-cut_off[1]] = np.arange(transition_length) / transition_length
                    overlap[-cut_off[1]:] = 1
                if transition_type == 'cut_off':
                    cut_off_pixel = int(np.ceil((before_last_pixel - swath_first_pixel) / 2))
                    overlap[-cut_off_pixel:] = 1

                swath_pixel_weights[:len(overlap)] = overlap

            if swath + 1 in swaths:
                after_slice_coordinates = [self.slice_coordinates[id] for id in np.ravel(np.argwhere(swaths == swath + 1))]
                after_first_pixel = np.max([coor.first_pixel for coor in after_slice_coordinates])

                if (swath_last_pixel - after_first_pixel) < 1:
                    raise ValueError('Not possible to concatenate using this transition type. Use coverage_cut_off instead.')

                overlap = np.zeros(swath_last_pixel - after_first_pixel)

                if transition_type == 'linear':
                    transition_length = len(overlap) - cut_off[1] * 2
                    overlap[cut_off[1]:-cut_off[1]] = np.flip(np.arange(transition_length)) / transition_length
                    overlap[:cut_off[1]] = 1
                if transition_type == 'cut_off':
                    cut_off_pixel = int(np.floor((swath_last_pixel - after_first_pixel) / 2))
                    overlap[:cut_off_pixel] = 1
                swath_pixel_weights[-len(overlap):] = overlap

            swath_slice_nums = slice_nums[swath_slice_ids]

            # Now apply same principal to the individual bursts but in line direction.
            for slice_num, slice_coor, slice_name in zip(swath_slice_nums, swath_slice_coordinates, swath_slice_names):

                # First get the weights in pixels
                slice_pixel_weights = np.zeros(slice_coor.shape[1])
                first_pix = swath_first_pixel - slice_coor.first_pixel
                slice_pixel_weights[first_pix: first_pix + len(swath_pixel_weights)] = swath_pixel_weights

                self.pixel_weights[slice_name] = slice_pixel_weights

                slice_line_weights = np.ones(slice_coor.shape[0])

                # Overlap if there is a burst before.
                if slice_num - 1 in swath_slice_nums:

                    before_coor = self.slice_coordinates[swath_slice_ids[list(swath_slice_nums).index(slice_num - 1)]]
                    before_last_line = before_coor.shape[0] + before_coor.first_line
                    overlap = np.zeros(before_last_line - slice_coor.first_line)

                    if transition_type == 'linear':
                        transition_length = len(overlap) - cut_off[0] * 2
                        overlap[cut_off[0]:-cut_off[0]] = np.arange(transition_length) / transition_length
                        overlap[-cut_off[0]:] = 1
                    if transition_type == 'cut_off':
                        cut_off_pixel = int(np.floor((before_last_line - slice_coor.first_line) / 2))
                        overlap[cut_off_pixel:] = 1
                    slice_line_weights[:len(overlap)] = overlap

                # Overlap if there is a burst after.
                if slice_num + 1 in swath_slice_nums:

                    after_coor = self.slice_coordinates[swath_slice_ids[list(swath_slice_nums).index(slice_num + 1)]]
                    after_first_line = after_coor.first_line
                    overlap = np.zeros(slice_coor.first_line + slice_coor.shape[0] - after_first_line)

                    if transition_type == 'linear':
                        transition_length = len(overlap) - cut_off[0] * 2
                        overlap[cut_off[0]:-cut_off[0]] = np.flip(np.arange(transition_length)) / transition_length
                        overlap[:cut_off[0]] = 1
                    if transition_type == 'cut_off':
                        cut_off_pixel = int(np.floor((slice_coor.first_line + slice_coor.shape[0] - after_first_line) / 2))
                        overlap[:cut_off_pixel] = 1
                    slice_line_weights[-len(overlap):] = overlap

                self.line_weights[slice_name] = slice_line_weights
