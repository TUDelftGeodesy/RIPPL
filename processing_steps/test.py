# This test is used to test a number of processing steps on one slave and master burst from Sentinel-1 data.
import os
import time
import copy
from joblib import Parallel, delayed
from doris_processing.image_data import ImageData

from doris_processing.orbit_dem_functions.srtm_download import SrtmDownload

from doris_processing.processing_steps.import_dem import CreateSrtmDem
from doris_processing.processing_steps.inverse_geocode import InverseGeocode
from doris_processing.processing_steps.radar_dem import RadarDem
from doris_processing.processing_steps.geocode import Geocode
from doris_processing.processing_steps.geometrical_coreg import GeometricalCoreg
from doris_processing.processing_steps.deramping_reramping import Deramp, Reramp
from doris_processing.processing_steps.resample import Resample
from doris_processing.processing_steps.earth_topo_phase import EarthTopoPhase
from doris_processing.processing_steps.azimuth_elevation_angle import AzimuthElevationAngle
from doris_processing.processing_steps.interfero import Interferogram
from doris_processing.processing_steps.coherence import Coherence
from doris_processing.processing_steps.concatenate import Concatenate

# Folders for master/slave and DEM processing (Last can be empty folder)
stack_folder = '/media/gert/Data/radar_datastacks/new_stack/'
master_date = '20161111'
slave_date = '20161105'
n_jobs = 8

slice_folders = next(os.walk('/media/gert/Data/radar_datastacks/new_stack/20161111'))[1]
folders = [os.path.join(stack_folder, master_date, slice) for slice in slice_folders]
masters = [ImageData(os.path.join(stack_folder, master_date, slice, 'info.res'), 'single') for slice in slice_folders]
slaves = [ImageData(os.path.join(stack_folder, slave_date, slice, 'info.res'), 'single') for slice in slice_folders]
srtm_folder = '/media/gert/Data/DEM/dem_processing'

import numpy as np

# Download SRTM data
self = SrtmDownload(srtm_folder, 'gertmulder', 'Radar2016', resolution='SRTM3', n_processes=8)
for master in masters:
    master.clean_memory()
    self(master)

# First create a DEM and do the geocoding for the master file.
start_time = time.time()

def inverse_geocode(master, master_folder):
    self = CreateSrtmDem(master_folder, dem_data_folder=srtm_folder, meta=master, resolution='SRTM3')
    self()
    self.create_output_files(master)
    self.save_to_disk()
    self = InverseGeocode(meta=master, resolution='SRTM3')
    self()
    self.create_output_files(master)
    self.save_to_disk()
    master.clean_memory()

    return master

masters = Parallel(n_jobs=8)(delayed(inverse_geocode)(master, master_folder) for master, master_folder in zip(masters, folders))

# Prepare geocoding output
for master in masters:
    sample, interval, buffer, coors, in_coors, out_coors = RadarDem.get_interval_coors(master)
    Geocode.add_meta_data(master, sample, coors, interval, buffer)
    Geocode.create_output_files(master, sample)
    RadarDem.add_meta_data(master, sample, coors, interval, buffer)
    RadarDem.create_output_files(master, sample)
    AzimuthElevationAngle.add_meta_data(master, sample, coors, interval, buffer, scatterer=True, orbit=True)
    AzimuthElevationAngle.create_output_files(master, sample, scatterer=True, orbit=True)
    master.clean_memory()

master_mat = []
slave_mat = []
i_mat = []
for n in range(n_jobs):
    master_mat.extend(copy.copy(masters))
    slave_mat.extend(copy.copy(slaves))
    i_mat.extend(list(np.ones(len(masters)).astype(np.int32) * n))

def geocoding(master, i):
    self = RadarDem(meta=master, resolution='SRTM3', s_lin=200 * i, lines=200, buf=5)
    self()
    self.save_to_disk()
    self = Geocode(meta=master, s_lin=200 * i, lines=200)
    self()
    self.save_to_disk()
    self = AzimuthElevationAngle(meta=master, orbit=True, scatterer=True, s_lin=200 * i, lines=200)
    self()
    self.save_to_disk()
    master.write()
    master.clean_memory()

    return master

master_mat = Parallel(n_jobs=8)(delayed(geocoding)(master, i) for master, i in zip(master_mat, i_mat))

# Time needed to do geocoding of master image.
print("--- %s seconds for geocoding ---" % (time.time() - start_time))

# Then coregister, deramp, resample, reramp and correct for topographic and earth reference phase
start_time = time.time()

for slave, master in zip(slave_mat[0:len(slaves)], master_mat[0:len(masters)]):
    # Preallocate the resampled data.
    EarthTopoPhase.add_meta_data(master, slave)
    EarthTopoPhase.create_output_files(slave)
    slave.clean_memory()
    master.clean_memory()

master_mat = []
slave_mat = []
i_mat = []
for n in range(n_jobs):
    master_mat.extend(copy.copy(masters))
    slave_mat.extend(copy.copy(slaves))
    i_mat.extend(list(np.ones(len(masters)).astype(np.int32) * n))

def resampling(master, slave, i):
    self = GeometricalCoreg(master, slave, s_lin=200 * i, lines=200)
    self()

    res = Resample(master, slave, warning=False, input_dat='crop')
    s_lin = res.s_lin_source
    s_pix = res.s_pix_source
    s_shape = res.shape_source

    self = Deramp(slave, s_lin=s_lin, s_pix=s_pix, lines=s_shape[0])
    self()
    self = Resample(master, slave, s_lin=200 * i, lines=200, input_dat='deramp')
    self()
    self = Reramp(master, slave, s_lin=200 * i, lines=200)
    self()
    self = EarthTopoPhase(master, slave, s_lin=200 * i, lines=200)
    self()
    self.save_to_disk()
    slave.write()
    slave.clean_memory()
    master.clean_memory()

    return master, slave

res_files = Parallel(n_jobs=8)(delayed(resampling)(master, slave, i) for master, slave, i in zip(master_mat, slave_mat, i_mat))

slave_mat = [res[1] for res in res_files]
slaves = slave_mat[:len(slaves)]


print("--- %s seconds for resampling ---" % (time.time() - start_time))


# Load the master and slave image
start_time = time.time()
ifgs = [Interferogram.create_meta_data(master, slave) for master, slave in zip(masters, slaves)]

concat_ifg = Concatenate(ifgs, multilook=[5, 20])

def create_ifg(master, slave, ifg, offset):
    self = Interferogram(master, slave, ifg, multilook=[5, 20], offset=offset)
    self()
    self = Coherence(master, slave, ifg, multilook=[5, 20], offset=offset)
    self()

    return ifg

ifgs = Parallel(n_jobs=8)(delayed(create_ifg)(master, slave, ifg, offset)
                          for master, slave, ifg, offset in zip(masters, slaves, ifgs, concat_ifg.offsets))

self = Concatenate(ifgs, multilook=[5, 20])
self('interferogram', 'Data')
self('coherence', 'Data')
self.concat.write()
self.concat.image_create_disk('interferogram', 'Data_ml_5_20')
self.concat.image_memory_to_disk('interferogram', 'Data_ml_5_20')
self.concat.image_create_disk('coherence', 'Data_ml_5_20')
self.concat.image_memory_to_disk('coherence', 'Data_ml_5_20')

print("--- %s seconds for creating interferogram ---" % (time.time() - start_time))

# Finally concatenate.





import numpy as np
import matplotlib.pyplot as plt

master.image_disk_to_memory('crop', 0, 0, master.data_disk['crop']['Data'].shape)

plt.figure(); plt.imshow(np.angle(master.data_memory['crop']['Data'][:,1000:2000] * slave_1.data_memory['earth_topo_phase']['Data'][:,1000:2000].conj()))
plt.figure(); plt.imshow(np.angle(master.data_memory['crop']['Data'][:,1000:2000] * slave_2.data_memory['earth_topo_phase']['Data'][:,1000:2000].conj()))
plt.figure(); plt.imshow(np.angle(slave_1.data_memory['earth_topo_phase']['Data'] * slave_2.data_memory['earth_topo_phase']['Data'].conj()))
plt.figure(); plt.imshow(np.angle(slave_1.data_memory['earth_topo_phase']['Data'][:,1000:2000] * master.data_memory['earth_topo_phase']['Data'][:,1000:2000].conj()))
plt.figure(); plt.imshow(np.angle(master.data_memory['crop']['Data'][:,1000:2000] * master.data_memory['earth_topo_phase']['Data'][:,1000:2000].conj()))

plt.figure(); plt.imshow(np.log(np.abs(master.data_memory['crop']['Data'][:,1000:2000] * slave_1.data_memory['earth_topo_phase']['Data'][:,1000:2000].conj())))
plt.figure(); plt.imshow(np.log(np.abs(master.data_memory['crop']['Data'][:,1000:2000] * slave_2.data_memory['earth_topo_phase']['Data'][:,1000:2000].conj())))
plt.figure(); plt.imshow(np.log(np.abs(slave_1.data_memory['earth_topo_phase']['Data'][:,1000:2000] * slave_2.data_memory['earth_topo_phase']['Data'][:,1000:2000].conj())))
plt.figure(); plt.imshow(np.log(np.abs(master.data_memory['crop']['Data'][:,1000:2000] * master.data_memory['reramp']['Data'][:,1000:2000].conj())))
