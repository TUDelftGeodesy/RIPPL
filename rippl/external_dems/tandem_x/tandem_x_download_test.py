
from rippl.external_dems.tandem_x.tandem_x_download import TandemXDownload, TandemXDownloadTile
from rippl.meta_data.stack import Stack

username = 'g.mulder-1@tudelft.nl'
password = 'Radar_2016'
resolution = 3
tdx_folder = '/mnt/fcf5fddd-48eb-445a-a9a6-bbbb3400ba42/DEM/TanDEM-X'
stack = '/mnt/f7b747c7-594a-44bb-a62a-a3bf2371d931/radar_datastacks/RIPPL_v2.0/Sentinel_1/Netherlands/asc_t088_test'

s1_stack = Stack(stack)
s1_stack.read_master_slice_list()
s1_stack.read_stack()
meta = s1_stack.slcs[s1_stack.master_date].data

download = TandemXDownload(tdx_folder, username, password, resolution, n_processes=1)
download(meta)
