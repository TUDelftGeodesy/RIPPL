"""
TEST

convert = ReadWriteShapes()
self = convert

# read in coordinates
coordinates = [(130.25, 32.28), (131.35, 32.28), (131.35, 33.24), (130.25, 33.24), (130.25, 32.28)]
convert(coordinates)

shapefile = '/Users/gertmulder/SAR_processing/GIS_database/layers/POLYGON.shp'

kml = '/Users/gertmulder/SAR_processing/GIS_database/layers/pol.kml'
convert.write_kml(kml)
convert.read_kml(kml)

geojson = '/Users/gertmulder/SAR_processing/GIS_database/layers/pol.json'
convert.write_geo_json(geojson)
json_dict = convert.write_geo_json()
convert(geojson)

# Finally save shapefile again
convert.write_shapefile(shapefile)
convert(shapefile)

"""

import ogr
import fiona
from fiona import collection
from shapely.geometry import Polygon, mapping
import shapely
from shapely.wkt import loads
import json
import os


class ReadWriteShapes():

    def __init__(self):

        # Loaded shape or shapes as shapely Polygons.
        self.shape = None         # type: Polygon
        self.shapes = []          # type: list(Polygon)
        self.shape_names = []     # type: list(str)

    def __call__(self, input_data):
        """
        This will read a shapefile/kml file based on the file extension. If it is a list of lists with coordinates,
        these will be loaded in.

        """

        if isinstance(input_data, str):

            if not os.path.exists(input_data):
                raise FileExistsError('File ' + input_data + ' does not exist!')

            if input_data.endswith('.shp'):
                self.read_shapefile(input_data)
                self.shape = self.shapes[0]
            elif input_data.endswith('.kml'):
                self.read_kml(input_data)
                self.shape = self.shapes[0]
            elif input_data.endswith('.json'):
                self.read_geo_json(input_data)
                self.shape = self.shapes[0]
            else:
                raise TypeError('File should either be a shapefile or a kml file.')

            for shape in self.shapes:
                if not isinstance(shape, Polygon):
                    raise TypeError('The input shapes for either shapefile or kml files should be Polygons! Points, '
                                    'Lines or Multipolygons cannot be used by the RIPPL package.')

        elif isinstance(input_data, list):
            self.read_coordinate_list(input_data)

        elif isinstance(input_data, dict):
            # We assume this is a loaded geojson file. (Or part of a .json metadata file)
            if not 'features' in input_data.keys():
                raise TypeError('If a dictionary is provided we assume it to be a geojson object, but it is missing '
                                'the features key!')

        else:
            raise TypeError('Input type should either be a .shp or .kml file or a list of coordinates.')

    def read_shapefile(self, shapefile):
        """
        Reads a .shp file and extracts one or more shapely polygons.

        """

        self.shapes = []
        with collection(shapefile, "r") as input_shape:
            for shape in input_shape:
                # only first shape
                self.shapes.append(shapely.geometry.shape(shape['geometry']))

    def read_kml(self, kml):
        """
        Reads a .kml file and extracts one or more shapely polygons.

        """

        driver = ogr.GetDriverByName('KML')
        dataSource = driver.Open(kml)
        layer = dataSource.GetLayer()

        self.shapes = []
        for feat in layer:
            self.shapes.append(loads(feat.geometry().ExportToWkt()))

    def read_geo_json(self, geojson):
        """
        Read from geojson
        1. If it is already loaded as a dictionary file
        2. If it given as a seperate geojson file.

        """

        if isinstance(geojson, str):
            with open(geojson, 'r') as json_file:
                json_data = json.load(json_file)
        else:
            json_data = geojson

        features = json_data["features"]
        self.shapes = [shapely.geometry.shape(feature['geometry']) for feature in features]

    def read_coordinate_list(self, coordinate_list):
        """
        Reads in two types of lists

        1. list of tuples wit lat/lon values
        2. List with a list of latitude values and one of longitude values

        """

        if not isinstance(coordinate_list, list):
            raise TypeError('Input should be a list of coordinate pairs or a list of latitudes + a list of longitudes.')

        if len(coordinate_list) == 2:
            self.shape = Polygon([[lat, lon] for lat, lon in zip(coordinate_list[0], coordinate_list[1])])
        else:
            self.shape = Polygon(coordinate_list)

        self.shapes = [self.shape]

    def write_shapefile(self, shapefile):
        """
        Write shape as a shapefile

        """

        schema = {'geometry': 'Polygon', 'properties': {'name': 'str', 'id': 'int'}}

        # Write a new Shapefile
        with fiona.open(shapefile, 'w', 'ESRI Shapefile', schema) as shape_dat:
            ## If there are multiple geometries, put the "for" loop here
            for id, shape in enumerate(self.shapes):
                if self.shape_names:
                    name = self.shape_names[id]
                else:
                    name = str(id)

                shape_dat.write({
                    'geometry': mapping(shape),
                    'properties': {'id': id, 'name': name},
                })

    def write_kml(self, kml):
        """
        Write the shapes as a kml file

        """

        driver = ogr.GetDriverByName('KML')
        ds = driver.CreateDataSource(kml)
        layer = ds.CreateLayer('', None, ogr.wkbPolygon)
        # Add one attribute
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn('name', ogr.OFTString))
        defn = layer.GetLayerDefn()

        for id, shape in enumerate(self.shapes):
            if self.shape_names:
                name = self.shape_names[id]
            else:
                name = str(id)

            # Create a new feature
            feat = ogr.Feature(defn)
            feat.SetField('id', id)
            feat.SetField('name', name)

            # Make a geometry, from Shapely object
            geom = ogr.CreateGeometryFromWkb(shape.wkb)
            feat.SetGeometry(geom)

            layer.CreateFeature(feat)
            feat = geom = None  # destroy these

        # Save and close everything
        ds = layer = feat = geom = None

    def write_geo_json(self, geojson=None):
        """
        Write as a geojson. This can be used as part of a json description.

        """

        feature_collection = {"type": "FeatureCollection",
                              "features": []}

        for id, shape in enumerate(self.shapes):
            if self.shape_names:
                name = self.shape_names[id]
            else:
                name = str(id)

            json_dict = {"type": "Feature", "properties":{'id': id, 'name': name}, "geometry":eval(ogr.CreateGeometryFromWkb(shape.wkb).ExportToJson())}
            feature_collection["features"].append(json_dict)

        if isinstance(geojson, str):
            with open(geojson, 'w') as json_file:
                json.dump(feature_collection, json_file)
        else:
            return feature_collection

    def extend_shape(self, buffer=0.1):
        """
        Create an extra buffer around the shape

        """

        self.shape = self.shape.buffer(buffer)
        shapes = []
        for shape in self.shapes:
            shapes.append(shape.buffer(buffer))
        self.shapes = shapes

    def simplify_shape(self, resolution=0.1):
        """
        Simplify shape

        """

        self.shape = self.shape.simplify(resolution)
        shapes = []
        for shape in self.shapes:
            shapes.append(shape.simplify(resolution))
        self.shapes = shapes
