import numpy as np
import pygrib
from scipy.interpolate.interpolate import RectBivariateSpline
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import copy

in_file = '/nobackup/users/mulder/grib_data/HA38_N25_201605300300_00300_GB/HA38_N25_201605300300_00300_GB'
start_time = datetime.datetime(year=int(in_file[-21:-17]), month=int(in_file[-17:-15]), day=int(in_file[-15:-13]),
                               hour=int(in_file[-13:-11]), minute=int(in_file[-11:-9]))
time = start_time
time_str = time.strftime('%Y%m%d%H%M')
time_step = 300
n_steps = 5
n_evaluate = 10
lat = 53

lat_step = 0.023 * 111000
lon_step = 0.037 * np.cos(lat / 180 * np.pi) * 111000

# Load wind
dat = pygrib.open(in_file)
layers = 65
v_size = 300
u_size = 300
size = (layers, v_size, u_size)
v_wind = np.zeros(size)
u_wind = np.zeros(size)
temperature = np.zeros(size)
spec_humidity = np.zeros(size)
cloud_water = np.zeros(size)
cloud_ice = np.zeros(size)

model_data = defaultdict()

v_interp = dict()
u_interp = dict()
t_interp = dict()
e_interp = dict()
w_interp = dict()
i_interp = dict()

# Select variables to save..
for var_name, var, interp in zip(['33', '34', '11', '51', '76', '58'],
                                 [u_wind, v_wind, temperature, spec_humidity, cloud_water, cloud_ice],
                                 [u_interp, v_interp, t_interp, e_interp, w_interp, i_interp]):
    vars = dat.select(parameterName=var_name)

    for variable, i in zip(vars, range(len(var))):
        var[i, :, :] = variable.values
        interp[str(i)] = RectBivariateSpline(range(u_size), range(v_size), variable.values, kx=1, ky=1)

# Prepare grid for wind movement.
new_u_grid, new_v_grid = np.meshgrid(range(v_size), range(u_size))
new_u_grid = np.tile(new_u_grid.astype('float32')[None, :, :], [layers, 1, 1])
new_v_grid = np.tile(new_v_grid.astype('float32')[None, :, :], [layers, 1, 1])
u_shift = u_wind / lon_step * time_step
v_shift = - v_wind / lat_step * time_step

model_data[time_str] = dict()
model_data[time_str]['Specific humidity'] = spec_humidity
model_data[time_str]['Temperature'] = temperature
model_data[time_str]['Specific cloud liquid water content'] = cloud_water
model_data[time_str]['Specific cloud ice water content'] = cloud_ice
model_data[time_str]['Wind_u'] = copy.deepcopy(new_u_grid)
model_data[time_str]['Wind_v'] = copy.deepcopy(new_v_grid)

# Now iterate over the new time_steps and evaluation time steps.
for evaluate in range(n_evaluate):

    for i in range(n_steps):
        for n in range(layers):
            u_shift[n, :, :] = u_interp[str(n)].ev(new_u_grid[n, :, :], new_v_grid[n, :, :]) * time_step / lon_step
            v_shift[n, :, :] = v_interp[str(n)].ev(new_u_grid[n, :, :], new_v_grid[n, :, :]) * time_step / lat_step

        new_u_grid += u_shift
        new_v_grid += v_shift

    time = time + datetime.timedelta(seconds=time_step * n_steps)
    time_str = time.strftime('%Y%m%d%H%M')

    model_data[time_str] = dict()
    model_data[time_str]['Wind_u'] = copy.deepcopy(new_u_grid)
    model_data[time_str]['Wind_v'] = copy.deepcopy(new_v_grid)
    model_data[time_str]['Specific humidity'] = np.zeros(size)
    model_data[time_str]['Temperature'] = np.zeros(size)
    model_data[time_str]['Specific cloud liquid water content'] = np.zeros(size)
    model_data[time_str]['Specific cloud ice water content'] = np.zeros(size)

    # Convert to new time_step.
    for n in range(layers):
        model_data[time_str]['Specific humidity'][n, :, :] = e_interp[str(n)].ev(new_u_grid[n, :, :], new_v_grid[n, :, :])
        model_data[time_str]['Temperature'][n, :, :] = t_interp[str(n)].ev(new_u_grid[n, :, :], new_v_grid[n, :, :])
        model_data[time_str]['Specific cloud liquid water content'][n, :, :] = w_interp[str(n)].ev(new_u_grid[n, :, :], new_v_grid[n, :, :])
        model_data[time_str]['Specific cloud ice water content'][n, :, :] = i_interp[str(n)].ev(new_u_grid[n, :, :], new_v_grid[n, :, :])

# Plot wind directions
keys = model_data.keys()

layer = 30
step = 20

#plt.figure()
#plt.quiver(u_shift[layer, ::step, ::step], v_shift[layer, ::step, ::step])
plt.figure()

wind_lines_u = np.zeros(((v_size / step) * (u_size / step), len(keys)))
wind_lines_v = np.zeros(((v_size / step) * (u_size / step), len(keys)))

for key, i in zip(sorted(keys), range(len(keys))):
    wind_lines_u[:, i] = np.ravel(model_data[key]['Wind_u'][layer, ::step, ::step])
    wind_lines_v[:, i] = np.ravel(model_data[key]['Wind_v'][layer, ::step, ::step])

for i in range(wind_lines_v.shape[0]):
    plt.plot(wind_lines_u[i, :], wind_lines_v[i, :])

    # Calculate the delay and plot time-series.