from setuptools import setup

setup(
    name='rippl',
    version='2.0',
    packages=['rippl'],
    url='https://bitbucket.org/grsradartudelft/rippl/',
    license='GNU LGPLv3',
    author='Gert Mulder',
    author_email='g.mulder-1@tudelft.nl',
    description='Radar Interferometric Parallel Processing Lab',
    install_requires=['numpy', 'scipy', 'scikit-image', 'matplotlib', 'gdal', 'fiona', 'shapely', 'requests', 'pyproj',
                      'lxml', 'utm', 'cartopy', 'jupyter'],
    classifiers=['Development Status :: 4 - Beta',
                   'License :: OSI Approved :: GNU LGPLv3',
                   'Programming Language :: Python :: 3'],
    keywords=['InSAR', 'Sentinel-1', 'SAR', 'Interferometry', 'Parallel processing', 'Stack processing', 'Remotes Sensing']
    )
