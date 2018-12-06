"""
This class creates a database of sentinel images stored in a certain folder.
It will list the images first by track and then by date.
Further it will give an oversight of the availability of the following files for every image:
1. Zipped data file
2. Unzipped data file
For the track, polarisation and dates of interest then
1. Load the manifest.safe [to filter for track/polarisation]
2. And compared with an area of interest filter [also obtained from the manifest.safe]
3. Then, from these images a list of bursts is compiled, with the lat/lon on the ground,
        x,y,z orbit position and azimuth time
This procedure will give us all the bursts we are interested in for a specific stack.

If there is a master burst file this can be loaded to compare with new bursts. Otherwise
you can create a new master burst file.

Finally we can check the overlap with the individual swaths too and unzip if needed.
"""

from shapely.geometry import Polygon, shape
import fiona
import copy
import datetime
from lxml import etree
import os
import zipfile
import warnings


class SentinelDatabase(object):

    def __init__(self):

        self.folder = []
        self.zip_images = list()
        self.unzip_images = list()

        self.image_info = dict()
        self.image_dummy = dict()

        self.image_dummy['coverage'] = []
        self.image_dummy['az_start_time'] = ''
        self.image_dummy['az_end_time'] = ''
        self.image_dummy['mode'] = ''
        self.image_dummy['orbit'] = ''
        self.image_dummy['polarisation'] = []
        self.image_dummy['product_type'] = ''
        self.image_dummy['direction'] = ''

        self.image_dummy['swath_tiff'] = ''
        self.image_dummy['swath_xml'] = ''

        self.shape = Polygon
        self.master_shape = Polygon
        self.selected_images = dict()

    def __call__(self, database_folder, shapefile, track_no, start_date='2010-01-01', end_date='2030-01-01', mode='IW', product_type='SLC', polarisation='VV'):
        # Run the different commands step by step

        # Select based on dates and read manifest files
        self.folder = database_folder
        self.index_folder()
        self.extract_manifest_files()
        self.select_date_track(track_no, start_date, end_date, mode, product_type, polarisation)

        # Read shape file and select images based on coverage.
        self.select_overlapping(shapefile, buffer=0.02)

    def index_folder(self):

        directory_list = list()
        for root, dirs, files in os.walk(self.folder):
            for name in dirs:
                if name.endswith('.SAFE'):
                    self.unzip_images.append(os.path.join(root, name))
                else:
                    directory_list.append(os.path.join(root, name))

            for f in files:
                if f.endswith('.zip'):
                    self.zip_images.append(os.path.join(root, f))

    def extract_manifest_files(self):

        for zip_image in self.zip_images:
            try:
                archive = zipfile.ZipFile(zip_image, 'r')
                if zip_image.endswith('.SAFE.zip'):
                    manifest = etree.parse(archive.open(os.path.join(os.path.basename(zip_image)[:-4], 'manifest.safe')))
                else:
                    manifest = etree.parse(archive.open(os.path.join(os.path.basename(zip_image)[:-4] + '.SAFE', 'manifest.safe')))
                zip_image = os.path.join('/vsizip/', zip_image)
                self.read_manifest_file(zip_image, manifest)
            except:
                print('Failed to read from: ' + zip_image)

        for unzip_image in self.unzip_images:
            manifest = etree.parse(os.path.join(unzip_image, 'manifest.safe'))
            self.read_manifest_file(unzip_image, manifest)

    def read_manifest_file(self, filename, manifest):
        # Read in coverage and other information from manifest.safe

        # Initialize this image
        im_dict = self.image_dummy

        # Information on image level.
        ns = {'gml': 'http://www.opengis.net/gml',
              'safe': 'http://www.esa.int/safe/sentinel-1.0',
              's1': 'http://www.esa.int/safe/sentinel-1.0/sentinel-1',
              's1sarl1': 'http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1'}
        coverage = [list(reversed([float(n) for n in m.split(',')])) for m in manifest.find('//gml:coordinates', ns).text.split()]
        im_dict['coverage'] = Polygon(coverage)
        im_dict['az_start_time'] = datetime.datetime.strptime(manifest.find('//safe:startTime', ns).text,
                                                              '%Y-%m-%dT%H:%M:%S.%f')
        im_dict['az_end_time'] = datetime.datetime.strptime(manifest.find('//safe:stopTime', ns).text,
                                                              '%Y-%m-%dT%H:%M:%S.%f')
        im_dict['mode'] = manifest.find('//s1sarl1:mode', ns).text
        im_dict['polarisation'] = [t.text for t in manifest.findall('//s1sarl1:transmitterReceiverPolarisation', ns)]
        im_dict['product_type'] = manifest.find('//s1sarl1:productType', ns).text
        im_dict['orbit'] = manifest.find('//safe:relativeOrbitNumber', ns).text
        im_dict['direction'] = manifest.find('//s1:pass', ns).text

        if filename.endswith('zip'):
            im_dict['path'] = filename

            # Give the .xml and .tiff files
            files = [m.attrib['href'] for m in manifest.xpath('//dataObject/byteStream/fileLocation')]
            if filename.endswith('.SAFE.zip'):
                im_dict['swath_tiff'] = [os.path.join(os.path.basename(filename[:-4]), f[2:]) for f in files if f.endswith('.tiff')]
                im_dict['swath_xml'] = [os.path.join(os.path.basename(filename[:-4]), f[2:]) for f in files if f.endswith('.xml') and len(f) < 90]
            else:
                im_dict['swath_tiff'] = [os.path.join(os.path.basename(filename[:-4] + '.SAFE'), f[2:]) for f in files if f.endswith('.tiff')]
                im_dict['swath_xml'] = [os.path.join(os.path.basename(filename[:-4] + '.SAFE'), f[2:]) for f in files if f.endswith('.xml') and len(f) < 90]
        else:
            im_dict['path'] = filename

            # Give the .xml and .tiff files
            files = [m.attrib['href'] for m in manifest.xpath('//dataObject/byteStream/fileLocation')]
            im_dict['swath_tiff'] = [f[2:] for f in files if f.endswith('.tiff')]
            im_dict['swath_xml'] = [f[2:] for f in files if f.endswith('.xml') and len(f) < 90]

        self.image_info[os.path.basename(filename)] = copy.deepcopy(im_dict)

    def select_date_track(self, track_no, start_date='2010-01-01', end_date='2030-01-01', mode='IW', product_type='SLC', polarisation='VV'):
        # Select all the images of interest

        if start_date:
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if end_date:
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

        for filename in self.image_info.keys():

            if track_no != int(self.image_info[filename]['orbit']) \
                    or mode != self.image_info[filename]['mode'] \
                    or product_type != self.image_info[filename]['product_type'] \
                    or polarisation not in self.image_info[filename]['polarisation']:
                continue
            if start_date:
                if start_date > self.image_info[filename]['az_start_time']:
                    continue
            if end_date:
                if end_date < self.image_info[filename]['az_start_time']:
                    continue

            self.selected_images[filename] = self.image_info[filename]

    def select_overlapping(self, shapefile, buffer=0.02):

        self.read_shapefile(shapefile, buffer=buffer)

        for key in list(self.selected_images.keys()):
            if not self.shape.intersects(self.selected_images[key]['coverage']):

                self.selected_images.pop(key)

    def read_shapefile(self, shapefile, buffer=0.02):
        # This function creates a shape to make a selection of usable bursts later on. Buffer around shape is in
        # degrees.

        try:
            # It should be a shape file. We always select the first shape.
            sh = next(fiona.open(shapefile))
            self.shape = shape(sh['geometry'])

            # Now we have the shape we add a buffer and simplify first to save computation time.
            self.shape = self.shape.simplify(buffer / 2)
            self.master_shape = self.shape.buffer(buffer)
            self.shape = self.shape.buffer(buffer + 0.05)
        except:
            warnings.warn('Unrecognized shape')
            return
