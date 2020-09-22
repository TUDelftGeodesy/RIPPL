"""
This class is used to plot data images.

The inputs are one image as the image to be plotted and optionally a second to define the transparency
The function generates a matplotlib object using cartopy so if needed you can make adjustements to the final plot
using standard matplotlib commands.

"""

import copy
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import cartopy.crs as ccrs
from shapely.geometry import Polygon, LinearRing
from shapely import speedups
speedups.disable()
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import RectBivariateSpline
import matplotlib as mpl
import scipy as sp

from rippl.meta_data.image_data import ImageData
from rippl.orbit_geometry.orbit_coordinates import OrbitCoordinates
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class PlotData(object):

    def __init__(self, data_in, coordinates=None, transparency_in=None, complex_plot='phase', complex_transparency='',
                 data_min_max=[], data_quantiles=[], data_scale='linear', data_cmap='viridis', data_cmap_midpoint=None,
                 transparency_min_max=[], transparency_quantiles=[], transparency_scale='linear',
                 lat_in=[], lon_in=[], max_pixels=10000000, margins=0, overwrite=False):

        """
        :param ImageData data_in:
        :param transparency_in:
        :param complex_plot:
        :param complex_transparency:
        :param data_max_min:
        :param data_quantiles:
        :param transparency_max_min:
        :param transparency_quantiles:
        """

        # information on plot
        self.figure = []
        self.main_axis = []
        self.inset_axis = []
        self.plot_main = []
        self.plot_dem = []
        self.plot_shape = []
        self.color_bar = []
        self.overwrite = overwrite

        # Information on dataset
        if not isinstance(data_in, ImageData) and not isinstance(data_in, np.ndarray):
            raise TypeError('data_in should be an ImageData object or np.ndarray')
        self.data_in = data_in
        self.min_max = data_min_max
        self.data_quantiles = data_quantiles
        self.data_scale = data_scale
        self.data_cmap = data_cmap
        self.data_cmap_midpoint = data_cmap_midpoint
        self.norm = ''
        self.complex_plot = complex_plot
        self.complex_transparency = complex_transparency

        # Information on transparency dataset
        self.transparency_in = transparency_in
        self.transparency_min_max = transparency_min_max
        self.transparency_quantiles = transparency_quantiles
        self.transparency_scale = transparency_scale

        # Coordinate system of in image
        if coordinates is None and isinstance(data_in, ImageData):
            self.coordinates = data_in.coordinates
        elif isinstance(coordinates, CoordinateSystem):
            self.coordinates = coordinates
        else:
            print("Either use an ImageData object as input or define the coordinate system.")

        self.crs = ccrs.Mercator()
        self.image_limits = []
        self.margins = margins
        self.lats = []
        self.lons = []

        # Max pixels to plot (otherwise plotting is not possible. Extension by factor 10 possible but will slow down
        # the plotting process significantly)
        self.max_pixels = max_pixels
        if isinstance(lat_in, ImageData):
            self.lat_in = lat_in
            self.lat_in.load_disk_data()
        else:
            self.lat_in = []
        if isinstance(lon_in, ImageData):
            self.lon_in = lon_in
            self.lon_in.load_disk_data()
        else:
            self.lon_in = []

        # Calculate the interval.
        total_pixels = self.coordinates.shape[0] * self.coordinates.shape[1]
        self.interval = int(np.ceil(np.sqrt(total_pixels / self.max_pixels)))

    def __call__(self, filename=''):

        self.image_filename(filename=filename)
        if not os.path.exists(self.filename) or self.overwrite:
            self.define_plot_limits()
            self.prepare_plotting_data()
            self.create_main_plot()
            self.plot_figure_data()
            self.add_axis_coordinates()
            self.create_inset()
            return True
        else:
            print('Filename already exists and overwrite is False')
            return False

    def prepare_plotting_data(self):
        """
        Load the data that is actually plotted. Both the real value and transparency.

        :return:
        """

        self.transparency_data = []

        if isinstance(self.data_in, ImageData):
            self.data_in.load_disk_data()
            plot_data = self.data_in.disk2memory(self.data_in.disk['data'][::self.interval, ::self.interval], self.data_in.dtype_disk)
        else:
            plot_data = self.data_in[::self.interval, ::self.interval]

        if plot_data.dtype == np.complex64:
            if self.complex_plot == 'phase':
                self.plot_data = np.angle(plot_data)
            elif self.complex_plot == 'amplitude':
                self.plot_data = np.abs(plot_data)
            else:
                raise TypeError('Only options phase and amplitude are possible for complex_plot')

            if self.complex_transparency == 'phase':
                self.transparency_data = np.angle(plot_data)
            elif self.complex_transparency == 'amplitude':
                self.transparency_data = np.abs(plot_data)
        else:
            self.plot_data = plot_data

        # Adjust scale of plot data
        self.plot_data[self.plot_data == 0] = np.nan
        self.plot_data = self.adjust_scale(self.plot_data, self.data_scale)

        # Find limits of plot data
        if len(self.min_max) == 2:
            pass
        else:
            if len(self.data_quantiles) != 2:
                self.data_quantiles = [0.1, 0.99]
            self.min_max = [np.nanquantile(self.plot_data, self.data_quantiles[0]),
                            np.nanquantile(self.plot_data, self.data_quantiles[1])]

        # Using the limits create the normalized colorscale.
        if self.data_cmap_midpoint != None:
            self.norm = MidpointNormalize(vmin=self.min_max[0], vmax=self.min_max[1], clip=False)

        # Get transparency information (if available)
        if isinstance(self.transparency_in, ImageData) or isinstance(self.transparency_in, np.ndarray):
            if isinstance(self.transparency_in, ImageData):
                self.transparency_in.load_disk_data()
                self.transparency_data = self.transparency_in.disk2memory(self.transparency_in.disk['data'][::self.interval, ::self.interval], self.data_in.dtype_disk)
            elif isinstance(self.transparency_in, np.ndarray):
                self.transparency_data = self.transparency_in[::self.interval, ::self.interval]

            if self.transparency_data.dtype == np.complex64:
                raise TypeError('Not possible to use complex data as transparency band!')

        if len(self.transparency_data) > 0:
            # Adjust scale of data
            self.transparency_data[self.transparency_data == 0] = np.nan
            self.transparency_data = self.adjust_scale(self.transparency_data, self.transparency_scale)

            # Find min/max values
            if len(self.transparency_min_max) == 2:
                pass
            else:
                if len(self.transparency_quantiles) != 2:
                    self.data_quantiles = [0.1, 0.99]
                self.transparency_min_max = [np.nanquantile(self.transparency_data, self.transparency_quantiles[0]),
                                np.nanquantile(self.transparency_data, self.transparency_quantiles[1])]

            # Scale to min/max values
            self.transparency_data = (self.transparency_data - self.transparency_min_max[0]) / np.diff(self.transparency_min_max)
            self.transparency_data[np.isnan(self.transparency_data)] = 0
            self.transparency_data[self.transparency_data < 0] = 0
            self.transparency_data[self.transparency_data > 1] = 1

    @staticmethod
    def adjust_scale(data, scale):
        """

        :param np.array data:
        :param str scale:
        :return:
        """

        if scale == 'linear':
            return data
        elif scale == 'log':
            return np.log10(data)
        elif scale == 'dB':
            return np.log10(data) * 10
        elif scale == 'square_dB':
            return np.log10(data**2) * 10
        else:
            raise TypeError('Scale type should be linear, log, dB or square_dB')

    def define_plot_limits(self):
        # This step creates a full lat/lon grid of the image. For radar coordinates this can be loaded using the geocoded
        # data, otherwise we will use a bilinear interpolation of about 100 points in the image.
        # Coordinates of projections will be calculated using pyproj library.

        coor = copy.deepcopy(self.coordinates)
        coor.shape = coor.shape + np.array([1, 1])

        if coor.grid_type == 'geographic':
            coor.lat0 = coor.lat0 - coor.dlat * self.interval / 2
            coor.lon0 = coor.lon0 - coor.dlon * self.interval / 2
            self.lats, self.lons = coor.create_latlon_grid(self.interval, self.interval)

        elif self.coordinates.grid_type == 'projection':
            coor.y0 = coor.y0 - coor.dy * self.interval / 2
            coor.x0 = coor.x0 - coor.dx * self.interval / 2
            x, y = self.coordinates.create_xy_grid(self.interval, self.interval)
            self.lats, self.lons = self.coordinates.proj2ell(x, y)

        elif self.coordinates.grid_type == 'radar_coordinates':
            if isinstance(self.lat_in, ImageData) and isinstance(self.lon_in, ImageData):
                self.lats = self.lat_in.disk['data'][::self.interval, ::self.interval]
                self.lons = self.lon_in.disk['data'][::self.interval, ::self.interval]
                size = self.lats.shape

                bbox = [-1, size[0] + 1, -1, size[1] + 1]
                interp_lats = RectBivariateSpline(np.arange(size[0]), np.arange(size[1]), self.lats, kx=1, ky=1, bbox=bbox)
                interp_lons = RectBivariateSpline(np.arange(size[0]), np.arange(size[1]), self.lons, kx=1, ky=1, bbox=bbox)

                self.lats = interp_lats(np.arange(size[0] + 1) - 0.5, np.arange(size[1] + 1) - 0.5)
                self.lats = interp_lons(np.arange(size[0] + 1) - 0.5, np.arange(size[1] + 1) - 0.5)

            else:
                # Interpolate using a bilinear grid.
                lines_array = np.arange(coor.first_line, coor.first_line + coor.shape[0] - self.interval / 2, self.interval)
                pixels_array = np.arange(coor.first_pixel, coor.first_pixel + coor.shape[1] - self.interval / 2, self.interval)

                # Calculated lines
                lines_calc_array = np.linspace(coor.first_line, coor.first_line + coor.shape[0], 10)
                pixels_calc_array = np.linspace(coor.first_pixel, coor.first_pixel + coor.shape[1], 10)
                lines_calc, pixels_calc = np.meshgrid(lines_calc_array, pixels_calc_array)

                orbit_in = OrbitCoordinates(coor)
                orbit_in.manual_line_pixel_height(np.ravel(lines_calc), np.ravel(pixels_calc), np.zeros(lines_calc.size))
                orbit_in.lph2xyz()
                orbit_in.xyz2ell()
                lat_bilinear = RectBivariateSpline(lines_calc_array, pixels_calc_array, np.reshape(orbit_in.lat, lines_calc.shape))
                lon_bilinear = RectBivariateSpline(lines_calc_array, pixels_calc_array, np.reshape(orbit_in.lon, pixels_calc.shape))

                self.lats = lat_bilinear(lines_array, pixels_array)
                self.lons = lon_bilinear(lines_array, pixels_array)

        else:
            raise TypeError('Grid type should be geographic, projection or radar_coordinates!')

        if not self.margins:
            self.margins = [0, 0]
        elif isinstance(self.margins, float) or isinstance(self.margins, int):
            self.margins = [self.margins, self.margins]

        # Define the image limits of the background image.
        self.lat_lim = [np.min(self.lats) - self.margins[1], np.max(self.lats) + self.margins[1]]
        self.lon_lim = [np.min(self.lons) - self.margins[0], np.max(self.lons) + self.margins[0]]
        self.image_limits = [self.lon_lim, self.lat_lim]
                             
    def create_main_plot(self):
        """
        Create base image, which is used as a base for the other images.

        :parameter bool margins: Turn on or off added margins of 0.1 the size of the full image

        :return:
        """

        ocean_10m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                                                edgecolor='face',
                                                facecolor=cfeature.COLORS['water'])
        land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                                edgecolor='face',
                                                facecolor=cfeature.COLORS['land'])

        self.figure = plt.figure(dpi=300)
        self.main_axis = self.figure.add_subplot(111, projection=self.crs)
        self.main_axis.coastlines(resolution='10m', zorder=10, alpha=0.5)
        self.main_axis.add_feature(ocean_10m, zorder=5)
        self.main_axis.add_feature(land_10m, zorder=1)
        self.main_axis.set_extent([self.lon_lim[0], self.lon_lim[1], self.lat_lim[0], self.lat_lim[1]])

    def create_inset(self, lat_extend=20, lon_extend=30):
        """

        :param fig:
        :param ax:
        :param crs:
        :return:
        """

        self.inset_axis = self.figure.add_axes([0.8, 0.8, 0.2, 0.2], projection=self.crs)
        self.inset_axis.set_extent([self.lon_lim[0] - (lon_extend / 2),
                        self.lon_lim[1] + (lon_extend / 2),
                        np.maximum(self.lat_lim[0] - (lat_extend / 2), -90),
                        np.minimum(self.lat_lim[1] + (lat_extend / 2), 90)])
        lat_shift = np.diff(self.lat_lim)[0] * 0.05
        lon_shift = np.diff(self.lon_lim)[0] * 0.05
        ring = LinearRing(list(zip(
            [self.lon_lim[0] - lon_shift, self.lon_lim[0] - lon_shift, self.lon_lim[1] + lon_shift,
             self.lon_lim[1] + lon_shift],
            [self.lat_lim[0] - lat_shift, self.lat_lim[1] + lat_shift, self.lat_lim[1] + lat_shift,
             self.lat_lim[0] - lat_shift])))
        self.inset_axis.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=0.75)
        self.inset_axis.coastlines()
        self.inset_axis.stock_img()

        p1 = self.main_axis.get_position()
        p2 = self.inset_axis.get_position()
        self.inset_axis.set_position([p1.x0, p1.y1 - p2.height, p2.width, p2.height])

    def add_axis_coordinates(self):
        """

        :param ax:
        :param crs:
        :return:
        """

        gl = self.main_axis.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.5)
        gl.xlabels_top = None
        gl.ylabels_right = None
        xgrid = np.arange(np.floor(self.lon_lim[0]) - 10, np.ceil(self.lon_lim[1]) + 10, 2)
        ygrid = np.arange(np.floor(np.min(self.lat_lim)) - 10, np.ceil(np.max(self.lat_lim)) + 10, 1)
        gl.xlocator = mticker.FixedLocator(xgrid.tolist())
        gl.ylocator = mticker.FixedLocator(ygrid.tolist())
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 11, 'color': 'black'}
        gl.ylabel_style = {'size': 11, 'color': 'black'}

    def plot_figure_data(self):
        """
        Plot the prepared data and transparency

        :return:
        """

        if len(self.transparency_data) > 0:
            # First create the color tuples
            if self.data_cmap_midpoint:
                rgb = self.norm(self.plot_data)
            else:
                rgb = Normalize(self.min_max[0], self.min_max[1], clip=True)(self.plot_data)
            rgb = plt.get_cmap(self.data_cmap)(rgb)

            rgb[:, :, -1] = self.transparency_data
            color_tuples = np.array([rgb[:, :, 0].flatten(), rgb[:, :, 1].flatten(),
                                     rgb[:, :, 2].flatten(), rgb[:, :, 3].flatten()]).transpose()

            self.plot_main = self.main_axis.pcolormesh(self.lons, self.lats, self.plot_data,
                                             color=color_tuples,
                                             zorder=6,
                                             transform=ccrs.PlateCarree())
        else:
            if self.data_cmap_midpoint == None:
                self.plot_main = self.main_axis.pcolormesh(self.lons, self.lats, self.plot_data,
                                                 cmap=self.data_cmap,
                                                 zorder=6,
                                                 transform=ccrs.PlateCarree(),
                                                 vmin=self.min_max[0],
                                                 vmax=self.min_max[1])
            else:
                self.plot_main = self.main_axis.pcolormesh(self.lons, self.lats, self.plot_data,
                                                 cmap=self.data_cmap,
                                                 zorder=6,
                                                 transform=ccrs.PlateCarree(),
                                                 norm=self.norm)

        # Plot colorbar
        self.color_bar = self.figure.colorbar(self.plot_main, shrink=0.8, orientation='vertical')

    def add_labels(self, title, value_name):
        """
        Add labels to plot. Here you can give a title to the figure and add information for the colorbar

        :param title:
        :param value_name:
        :return:
        """

        self.main_axis.set_title(title)
        self.color_bar.set_label(value_name)

    def add_shape(self, shape, linewidth=1, color='black'):
        """
        Add a shape to the image (For example the initial area of interest of an applied image mask.)

        :param Polygon shape:
        :param line_thickness:
        :param color:
        :return:
        """

        self.plot_shape = self.main_axis.add_geometries([shape], edgecolor=color, facecolor="none", zorder=20,
                              linewidth=linewidth, crs=ccrs.PlateCarree())

    def add_background_dem(self, dem_data, colorbar=False):
        """
        Add background DEM data instead of the standard green. Oceans and water will stay blue.

        :param dem_data:
        :return:
        """

        if not isinstance(dem_data, ImageData):
            raise TypeError('DEM data should be an ImageData object')
        else:
            self.dem_data = dem_data

        self.dem_data.load_disk_data()
        self.dem = self.dem_data.disk2memory(dem_data.disk['data'][::self.interval, ::self.interval], self.dem_data.dtype)

        self.plot_dem = self.main_axis.pcolormesh(self.lons, self.lats, self.dem, cmap='terrain', zorder=2, transform=ccrs.PlateCarree())
        if colorbar:
            self.color_bar = plt.colorbar(self.plot_dem, shrink=0.8)

    def image_filename(self, filename=''):
        """
        Check image filename if file is going to be saved.

        :param filename:
        :return:
        """

        if len(filename) > 0:
            if not os.path.exists(os.path.dirname(filename)):
                print("Path to " + filename + ' does not exist!')
            else:
                self.filename = filename
        elif isinstance(self.data_in, ImageData):
            # Get filepath from ImageData object.
            basename = os.path.basename(self.data_in.file_path)[:-4] + '.png'
            folder_name = os.path.basename(os.path.dirname(self.data_in.file_path))
            if folder_name.startswith('slice'):
                date_name = os.path.dirname(os.path.dirname(self.data_in.file_path))
                self.filename = date_name + '_' + folder_name + '_' + basename
            else:
                date_name = os.path.dirname(self.data_in.file_path)
                self.filename = date_name + '_' + basename

    def save_image(self):
        """
        Save figure to disk.

        :return:
        """

        self.figure.savefig(self.filename, bbox_inches='tight', pad_inches=0.2)

    def plot_image(self):
        """
        Plot image

        :return:
        """

        plt.show()

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return sp.ma.masked_array(np.ma.masked_invalid(sp.interp(value, x, y)))
