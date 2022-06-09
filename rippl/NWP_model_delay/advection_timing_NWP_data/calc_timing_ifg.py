from rippl.meta_data.stack import Stack
import numpy as np

resolution = '200'
data_folder = '/media/gert/SAR_data/radar_data_stacks/Sentinel-1/Benelux_track_37/'
ifg_filename = 'unwrapped_VV@proj_oblique_mercator_' + resolution + '_' + resolution + '_in_coor_proj_oblique_mercator_' + resolution + '_' + resolution + '.raw'
filenames = []
for t in np.arange(-12, 13):

    if t != 0:
        time_str = '_' + str(t * 5) + '_minutes'
    else:
        time_str = ''
    filenames.append('harmonie_time_corrected_aps' + time_str + '@proj_oblique_mercator_' + resolution + '_' + resolution + '_in_coor_proj_oblique_mercator_2500_2500.raw')

# Load the data
shape = (2849, 1299)

# ifg date
date_1 = '20160506'
date_2 = '20160518'

data_ifg = np.memmap(data_folder  + date_1 + '_' + date_2 + '/' + ifg_filename, dtype=np.float32, mode='r+', shape=shape)
data_harmonie1 = []
data_harmonie2 = []
for filename in filenames:
    data_harmonie1.append(np.memmap(data_folder + date_1 + '/' + filename, dtype=np.float32, mode='r+', shape=shape))
    data_harmonie2.append(np.memmap(data_folder + date_2 + '/' + filename, dtype=np.float32, mode='r+', shape=shape))

# Plotting
import matplotlib.pyplot as plt

for n in range(len(data_harmonie1)):
    print(filenames[n])

    plt.figure()
    plt.imshow(data_harmonie2[n] - data_harmonie1[n])
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(data_ifg / np.pi * 0.056)
    plt.colorbar()
    plt.show()
