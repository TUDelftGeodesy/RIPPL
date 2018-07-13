
import sys, os
import datetime
# This script is used to project_functions all data from the ESA website to the cluster

# First load the project_functions scripts.
sys.path.extend(['/home/gert/software/phd_code/Gert/doris_processing/sentinel'])
from sentinel_download import sentinel_available, sentinel_download, download_orbits, sentinel_check_validity

ROI = '/media/gert/Data/shapes/netherlands/netherland.shp'
orbit_folder = '/media/gert/Data/orbit'
destination_folder = '/media/gert/Data/radar_database/sentinel-1'
level = 'L1'
sensor_mode = 'IW'
product = 'SLC'
polarisation = 'VV VH'
user = 'fjvanleijen'
password = 'stevin01'
orbit_direction = ''

tracks = ['88', '110', '37', '161']
products = []
links = []
dates = []

start_day = '2014-01-01'
end_day = datetime.datetime.now().strftime('%Y-%m-%d')

for track in tracks:
    prod, l, d = sentinel_available(start_day=start_day, end_day=end_day, ROI=ROI, polarisation=polarisation,
                                    sensor_mode=sensor_mode, track=track, orbit_direction=orbit_direction, level=level,
                                    product=product, user=user, password=password)
    products.extend(prod)
    links.extend(l)
    dates.extend(d)

for i in range(1000):
    sentinel_download(products, project_folder='', destination_folder=destination_folder, user=user, password=password)
    download_orbits(pages=5, precise_folder=os.path.join(orbit_folder,'sentinel-1'), restituted_folder=os.path.join(orbit_folder,'sentinel-1'))

    # Finally do a cleanup of all corrupted data.
    invalid_files, valid_files = sentinel_check_validity(products, destination_folder=destination_folder, user=user, password=password, remove=False)
