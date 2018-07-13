

import getpass
import os
# from copy_sentinel_data import find_paths

server = True
host = 'hpc03.tudelft.net'
username = 'gertmulder'
type = 'specific_burst'
network = False
convert = True

filenames = ['slave_rsmp_reramped.raw', 'slave.res', 'ifgs.res', 'master.res']
base_cluster = '/home/gertmulder/datastacks'
base_folder = '/media/gert/Data/radar_data/'
tracks = ['netherlands/nl_full_t037', 'netherlands/nl_full_t110', 'netherlands/nl_full_t088_vv']
burst_names = ['swath_3_burst_11', 'swath_2_burst_7', 'swath_1_burst_7']

password = getpass.getpass("Enter your password")

for specific_burst, track in zip(burst_names, tracks):
    stack_folder = os.path.join(base_cluster, track)
    output_folder = os.path.join(base_folder, track)

    for filename in filenames:
        find_paths(stack_folder, output_folder, filename, server, host, username, password, type, network, convert, specific_burst)
