# Code to read a sentinel data file using gdal

import gdal
import numpy as np
import os


def sentinel_read_data(path_tiff, s_pix, s_lin, size):

    if 'zip' in path_tiff:
        src_ds = gdal.Open('/vsizip/' + path_tiff, gdal.GA_ReadOnly)
    else:
        src_ds = gdal.Open(path_tiff, gdal.GA_ReadOnly)
    if src_ds is None:
        print 'Unable to open ' + path_tiff
        return

    band = src_ds.GetRasterBand(1)
    dat = band.ReadAsArray(s_pix - 1, s_lin - 1, size[1], size[0])

    return dat

def write_sentinel_burst(stack_folder, slice, number, pol, swath_no, date):
    slice_code = 'slice_' + str(number) + '_swath_' + str(swath_no) + '_' + pol
    folder_date = date[:4] + date[5:7] + date[8:]

    if slice == 'NoData':
        print('Image already loaded. Skipping image ' + slice_code + ' at ' + date)
        return []

    folder = os.path.join(stack_folder, folder_date, slice_code)

    if os.path.exists(folder):
        print('Image already loaded. Skipping image ' + slice_code + ' at ' + date)
        return
    else:
        os.makedirs(folder)

    # Find the pixels we are interested in from the .tiff file.
    first_line = int(slice.processes['crop']['crop_first_line (w.r.t. tiff_image)'])
    lines = int(slice.processes['crop']['crop_lines'])
    first_pixel = int(slice.processes['crop']['crop_first_pixel'])
    pixels = int(slice.processes['crop']['crop_pixels'])

    data_path = slice.processes['readfiles']['Datafile']
    slice_file = os.path.join(folder, 'crop.raw')
    slice_res = os.path.join(folder, 'info.res')

    # Save datafile
    data = sentinel_read_data(data_path, first_pixel, first_line, [lines, pixels])

    if len(data) == 0:
        print('Unable to load .tiff file. Failed initialize ' + slice_code + ' at ' + date)
    else:
        data_file = np.memmap(slice_file, dtype=np.dtype([('re', np.int16), ('im', np.int16)]), shape=(lines, pixels),
                              mode='w+')
        data_file[:, :] = data.view(np.float32).astype(np.int16).view(np.dtype([('re', np.int16), ('im', np.int16)]))
        data_file.flush()

        # Save resfile
        slice.write(new_filename=slice_res)
        print('Finished initialization of ' + slice_code + ' at ' + date)
