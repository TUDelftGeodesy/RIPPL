from setuptools import setup

setup(
    name='rippl',
    version='3.7',
    packages=['rippl', 'rippl.meta_data', 'rippl.resampling', 'rippl.SAR_sensors', 'rippl.SAR_sensors.radarsat',
              'rippl.SAR_sensors.sentinel', 'rippl.external_dems', 'rippl.external_dems.srtm',
              'rippl.external_dems.tandem_x', 'rippl.orbit_geometry', 'rippl.NWP_simulations',
              'rippl.NWP_simulations.ECMWF', 'rippl.NWP_simulations.harmonie', 'rippl.processing_steps',
              'rippl.processing_steps_old', 'examples', 'examples.old'],
    url='',
    license='',
    author='Gert Mulder',
    author_email='g.mulder-1@tudelft.nl',
    description='Radar Interferometric Parallel Processing Lab',
    install_requires=['numpy', 'scipy', 'matplotlib', 'gdal', 'fiona', 'shapely', 'os', 'urllib', 'pickle', 'zipfile',
                      'requests', 'datetime', 'shutil']
)
