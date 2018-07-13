from processing_steps.import_dem import CreateSrtmDem
from processing_steps.inverse_geocode import InverseGeocode
from processing_steps.radar_dem import RadarDem
from processing_steps.geocode import Geocode
from processing_steps.geometrical_coreg import GeometricalCoreg
from processing_steps.deramping_reramping import Deramp, Reramp
from processing_steps.resample import Resample
from processing_steps.earth_topo_phase import EarthTopoPhase
from processing_steps.azimuth_elevation_angle import AzimuthElevationAngle
from processing_steps.interfero import Interfero
from processing_steps.coherence import Coherence
from processing_steps.unwrap import Unwrap
import os

def resampling(master, slave, block, lines):
    im = GeometricalCoreg(master, slave, s_lin=lines * block, lines=lines)
    im()

    res = Resample(master, slave, s_lin=lines * block, lines=lines, warning=False, input_dat='crop')
    try:
        s_lin = res.s_lin_source
        s_pix = 0
        s_shape = res.shape_source
    except:
        print('Failed to load original image crop. Required region not known.')
        return master, slave

    im = Deramp(slave, s_lin=s_lin, s_pix=s_pix, lines=s_shape[0])
    im()
    im = Resample(master, slave, s_lin=lines * block, lines=lines, input_dat='deramp')
    im()
    im = Reramp(master, slave, s_lin=lines * block, lines=lines)
    im()
    im = EarthTopoPhase(master, slave, s_lin=lines * block, lines=lines)
    im()
    im.save_to_disk()
    slave.write()
    slave.clean_memory()
    master.clean_memory()

    print('Finished resampling ' + slave.res_path + ' lines ' + str(lines * block) + ' to ' + str(lines * (block + 1)))

    return master, slave

def geocoding(slice, block, lines, interval, buffer):
    im = RadarDem(meta=slice, resolution='SRTM3', s_lin=lines * block, lines=lines, buf=5, buffer=buffer, interval=interval)
    im()
    im.save_to_disk()
    im = Geocode(meta=slice, s_lin=lines * block, lines=lines, buffer=buffer, interval=interval)
    im()
    im.save_to_disk()
    im = AzimuthElevationAngle(meta=slice, orbit=True, scatterer=True, s_lin=lines * block, lines=lines, buffer=buffer, interval=interval)
    im()
    im.save_to_disk()
    slice.write()
    slice.clean_memory()

    print('Finished geocoding ' + slice.res_path + ' lines ' + str(lines * block) + ' to ' + str(lines * (block + 1)))

    return slice

def create_dem(slice, dem_folder):
    im = CreateSrtmDem(slice.folder, dem_data_folder=dem_folder, meta=slice, resolution='SRTM3')
    im()
    im.create_output_files(slice)
    im.save_to_disk()
    slice.write()
    slice.clean_memory()

    print('Finished importing SRTM data ' + slice.res_path)

    return slice

def create_dem_lines(slice, dem_folder):
    im = CreateSrtmDem(slice.folder, dem_data_folder=dem_folder, meta=slice, resolution='SRTM3')
    im()
    im.create_output_files(slice)
    im.save_to_disk()
    slice.write()
    slice.clean_memory()

    print('Finished importing SRTM data ' + slice.res_path)

    return slice, im.dem.shape[0]

def inverse_geocode(slice, block, lines):
    im = InverseGeocode(meta=slice, resolution='SRTM3', s_lin=lines * block, lines=lines)
    im()
    im.save_to_disk()
    slice.write()
    slice.clean_memory()

    print('Finished inverse geocoding ' + slice.res_path)

    return slice

def simulate_aps(slice, block, lines, ecmwf=False, ecmwf_type='era5', ecmwf_folder='',
                 harmonie=False, harmonie_type='h_38', harmonie_folder=''):

    im = InverseGeocode(meta=slice, resolution='SRTM3', s_lin=lines * block, lines=lines)
    im()
    im.save_to_disk(slice)
    slice.write()
    slice.clean_memory()

    print('Finished inverse geocoding ' + slice.res_path)

    return slice

def create_ifg(master, slave, ifg, multilook, oversampling, offset):
    im = Interfero(master, slave, ifg, multilook=multilook, oversampling=oversampling, offset=offset)
    im()
    im = Coherence(master, slave, ifg, multilook=multilook, oversampling=oversampling, offset=offset)
    im()
    master.write()
    slave.write()
    master.clean_memory()
    slave.clean_memory()

    print('Finished creating ifg of ' + slave.res_path + ' and ' + master.res_path)

    return ifg


def unwrap(ifg, multilook):
    im = Unwrap(ifg, multilook=multilook)
    im()
    ifg.clean_memory()

    return ifg


def structure_function(meta, ifg=True, s_lin=0, s_pix=0, lines=0, input_step='unwrap', multilook='', offset=''):
    im = StructureFunctions(meta, ifg, s_lin, s_pix, lines, input_step, multilook, offset)
    im()
    meta.clean_memory()

    return meta
