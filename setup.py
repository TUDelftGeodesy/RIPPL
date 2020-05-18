from setuptools import setup

setup(
    name='rippl',
    version='2.0',
    packages=['rippl'],
    url='',
    license='',
    author='Gert Mulder',
    author_email='g.mulder-1@tudelft.nl',
    description='Radar Interferometric Parallel Processing Lab',
    install_requires=['numpy', 'scipy', 'scikit-image', 'matplotlib', 'gdal', 'fiona', 'shapely', 'requests', 'pyproj', 'lxml']
)
