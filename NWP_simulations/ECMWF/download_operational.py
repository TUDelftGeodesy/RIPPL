import datetime
import calendar
from collections import OrderedDict
import os
import pygrib
import numpy as np

output_folder = '/nobackup/users/mulder/data/ecmwf/operational'

outputs_atmo = []
outputs_surf = []
request_script = []
outputs_txt = []
month_dates = []
start_month = '2014-09'
end_month = '2018-01'

start = datetime.datetime.strptime(start_month, '%Y-%m')
end = datetime.datetime.strptime(end_month, '%Y-%m')

date = start
while date <= end:
    days = calendar.monthrange(date.year, date.month)
    i = 1
    request_date = []
    while i < days[1]:
        day_1 = i
        day_2 = np.minimum(i + 4, days[1])
        request_date.append((date.strftime('%Y-%m-') + str(day_1).zfill(2) + '/to/' +
                        date.strftime('%Y-%m-') + str(day_2).zfill(2)))
        i += 5

    month_dates.append(request_date)
    outputs_atmo.append('ECMWF_operational_' + date.strftime('%Y%m') + '_atmosphere.grb')
    outputs_surf.append('ECMWF_operational_' + date.strftime('%Y%m') + '_surface.grb')
    date += datetime.timedelta(days=int(days[1]))

class_str = 'od'
grid = '0.1/0.1'
area = '56.0/-2.0/45.0/12.0'

levels = 137
level_list = ''
for l in range(1, levels + 1):
    level_list += str(l) + '/'
level_list = level_list[:-1]

t_step = 6
t_list = ''
for t in range(0, 24, t_step):
    t_list += str(t).zfill(2) + ':00:00' + '/'
t_list = t_list[:-1]

n = 6
a = 1

for month_date, output_atmo, output_surf in zip(month_dates, outputs_atmo, outputs_surf):

    if n == 6:
        if a > 1:
            f.close()

        f = open(os.path.join(output_folder, 'run_ecmwf_mars_request_' + str(a)), 'w')
        f.write('#!/bin/ksh\n')
        f.write('\n')
        f.write('export PATH=$PATH:.:$HOME/bin\n')
        f.write('export REMOTE=download_ecmwf@genericSftp\n')
        f.write('export GATEWAY=ecaccess.knmi.nl\n')
        f.write('export MARS_MULTITARGET_STRICT_FORMAT=1\n')
        n = 0
        a += 1
    else:
        n += 1

    mars_atmo = OrderedDict({
        "class": class_str,
        "date": month_date[0],
        "expver": "1",
        "grid": grid,
        "levelist": level_list,
        "levtype": "ml",
        "param": "130.128/133.128/246.128/247.128",
        "step": "0",
        "stream": "oper",
        "time": t_list,
        "type": "an",
        "area": area,
        "target": output_atmo,
    })

    mars_surface = OrderedDict({
        "class": class_str,
        "date": month_date[0],
        "expver": "1",
        "grid": grid,
        "levelist": '1',
        "levtype": "ml",
        "param": "129.128/152.128",
        "step": "0",
        "stream": "oper",
        "time": t_list,
        "type": "an",
        "area": area,
        "target": output_surf,
    })

    f.write('\n')
    f.write('mars << +++EOI\n')
    f.write('\tretrieve,\n')
    keys = mars_atmo.keys()
    for key, i in zip(keys, range(len(keys))):
        if i == len(keys) - 1:
            f.write('\t\t' + key + '=' + mars_atmo[key] + '\n')
        else:
            f.write('\t\t' + key + '=' + mars_atmo[key] + ',\n')
    for m in month_date[1:]:
        f.write('\tretrieve,\n')
        f.write('\t\t' + 'date=' + m + '\n')
    f.write('+++EOI\n')

    f.write('')
    f.write('mars << +++EOI\n')
    f.write('\tretrieve,\n')
    keys = mars_surface.keys()
    for key, i in zip(keys, range(len(keys))):
        if i == len(keys) - 1:
            f.write('\t\t' + key + '=' + mars_surface[key] + '\n')
        else:
            f.write('\t\t' + key + '=' + mars_surface[key] + ',\n')
    for m in month_date[1:]:
        f.write('\tretrieve,\n')
        f.write('\t\t' + 'date=' + m + '\n')
    f.write('+++EOI\n')

f.close()

f = open(os.path.join(output_folder, 'download_ecmwf_data'), 'w')
f.write('#!/bin/ksh\n')
f.write('\n')
f.write('export PATH=$PATH:.:$HOME/bin\n')
f.write('export REMOTE=download_ecmwf@genericSftp\n')
f.write('export GATEWAY=ecaccess.knmi.nl\n')
f.write('export MARS_MULTITARGET_STRICT_FORMAT=1\n')

f.write('for file in ECMWF_operational_*.grb; do\n')
f.write('\tectrans -overwrite -source $file\n')
f.write('done\n')
f.close()

# Test the downloaded datasets
weather_data_file = os.path.join('/nobackup/users/mulder/data/ecmwf/operational', 'ECMWF_operational_201409')
model_data = dict()
levels = 137
hour = 06
date = 20140920

print('Loading data file ' + weather_data_file + '_atmosphere.grb')
index_a = pygrib.index(weather_data_file + '_atmosphere.grb', 'name', 'level', 'hour', 'dataDate')
print('Loading data file ' + weather_data_file + '_surface.grb')
index_s = pygrib.index(weather_data_file + '_surface.grb', 'name', 'hour', 'dataDate')

dat_types = ['Specific humidity', 'Temperature', 'Specific cloud liquid water content',
             'Specific cloud ice water content']

var = index_a(name='Temperature', level=1, hour=hour, dataDate=date)[0]
latitudes = np.unique(var['latitudes'])
longitudes = np.unique(var['longitudes'])
model_data['latitudes'] = latitudes
model_data['longitudes'] = longitudes
dat_shape = (levels, len(latitudes), len(longitudes))

for dat_type in dat_types:

    model_data[dat_type] = np.zeros(shape=dat_shape)

    for level in range(1, levels + 1):
        var = index_a(name=dat_type, level=level, hour=hour, dataDate=date)[0]
        model_data[dat_type][level - 1, :, :] = var.values

# Calculate pressure and heights for model levels.
geo = index_s.select(name='Geopotential', hour=hour, dataDate=date)[0]
geo_h = geo.values / 9.80665

log_p = index_s.select(name='Logarithm of surface pressure', hour=hour, dataDate=date)[0]
geo_p = np.exp(log_p.values)

log_p = index_s.select(name='Surface pressure', hour=hour, dataDate=date)[0]
geo_p = np.exp(log_p.values)



