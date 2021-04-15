# Code to read a sentinel data file using gdal

from osgeo import gdal
import numpy as np
import os
from rippl.meta_data.image_processing_data import ImageProcessingData


def sentinel_read_data(path_tiff, s_pix, s_lin, size):

    if 'zip' in path_tiff:
        os.environ['CPL_ZIP_ENCODING'] = 'UTF-8'
        if os.name == 'nt':
            path, zip_path = path_tiff.split('.zip')
            path_tiff = path + '.zip' + zip_path.replace('\\', '/')
        src_ds = gdal.Open('/vsizip/' + path_tiff, gdal.GA_ReadOnly)
    else:
        src_ds = gdal.Open(path_tiff, gdal.GA_ReadOnly)
    if src_ds is None:
        print('Unable to open ' + path_tiff)
        return

    band = src_ds.GetRasterBand(1)
    dat = band.ReadAsArray(int(s_pix) + 1, int(s_lin) + 1, int(size[1]), int(size[0]))
    del src_ds

    return dat

def write_sentinel_burst(input):

    stack_folder = input[0]
    slice = input[1]
    number = input[2]
    swath_no = input[3]
    date = input[4]
    [no, n] = input[5]

    slice_code = 'slice_' + str(number) + '_swath_' + str(swath_no)
    folder_date = date[:4] + date[5:7] + date[8:]

    print('Loading from original SLC, burst ' + str(no) + ' out of ' + str(n))

    if slice == 'NoData':
        print('Image already loaded. Skipping image ' + slice_code + ' at ' + date)
        return []

    folder = os.path.join(stack_folder, folder_date, slice_code)

    # Find the pixels we are interested in from the .tiff file.
    pol = slice.readfiles['original'].json_dict['Polarisation']
    crop_key = [key for key in slice.processes['crop'].keys() if pol in key][0]
    crop_coor = slice.processes['crop'][crop_key].coordinates

    first_line = int(slice.readfiles['original'].json_dict['First_line (w.r.t. tiff_image)']) + crop_coor.orig_first_line
    lines = crop_coor.shape[0]
    first_pixel = crop_coor.orig_first_pixel
    pixels = crop_coor.shape[1]

    data_path = slice.readfiles['original'].json_dict['Datafile']
    slice_json = os.path.join(folder, 'info.json')

    if not os.path.exists(folder):
        os.makedirs(folder)

    if os.path.exists(slice_json):
        # In the case the processing already exists but for a different polarisation.
        new_slice = ImageProcessingData(os.path.dirname(slice_json))
        new_slice.readfiles['original'].json_dict['Polarisation'] += (',' + pol)
        new_slice.add_process(slice.processes['crop'][crop_key])
    else:
        new_slice = slice

    new_slice.processes_data['crop'][crop_key].images['crop'].folder = folder
    new_slice.processes_data['crop'][crop_key].images['crop'].create_file_name()
    slice_file = new_slice.processes_data['crop'][crop_key].images['crop'].file_path

    if os.path.exists(slice_json) and os.path.exists(slice_file):
        print('Image already loaded. Skipping image ' + slice_code + ' with polarisation ' + pol + ' at ' + date)
        return

    # Save datafile
    data = sentinel_read_data(data_path, first_pixel, first_line, [lines, pixels])

    if len(data) == 0:
        print('Unable to load .tiff file. Failed initialize ' + slice_code + ' with polarisation ' + pol + ' at ' + date)
    else:


        data_file = np.memmap(slice_file, dtype=np.dtype([('re', np.int16), ('im', np.int16)]), shape=(lines, pixels),
                              mode='w+')
        data_file[:, :] = data.view(np.float32).astype(np.int16).view(np.dtype([('re', np.int16), ('im', np.int16)]))
        data_file.flush()

        # Save resfile
        new_slice.processes_data['crop'][crop_key].images['crop'].check_data_disk_valid()
        new_slice.meta.save_json(json_path=slice_json)
        print('Finished initialization of ' + slice_code + ' with polarisation ' + pol + ' at ' + date)

