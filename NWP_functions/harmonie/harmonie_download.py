
import os
import sys, tarfile
import datetime
import pygrib

# Download data from mos to my own disk or from the ftp server if the data is recent.


# List the needed folders
base_folder = '/nobackup/users/mulder/sar_results'
grib_folder = '/nobackup/users/mulder/grib_data'
tracks = ['netherlands/nl_full_t110', 'netherlands/nl_full_t088_vv',
          'netherlands/nl_full_t088_vh', 'europe/nl_france_t037', 'netherlands/nl_full_t037']
analysis_times = ['06', '15', '15', '06', '06']
forecast_times = ['0600_00000_GB', '1500_00200_GB', '1500_00200_GB', '0600_00000_GB', '0600_00000_GB']
needed_vars = ['Cloud water', 'Specific humidity', 'Temperature', 'Pressure_3d', 'surface_pressure']
var_num = ['76', '51', '11', '212', '1']

for track, analysis_time, forecast_time in zip(tracks, analysis_times, forecast_times):
    # Now find the different dates and project_functions some additional files.

    if os.path.exists(os.path.join(base_folder, track)):
        folders = next(os.walk(os.path.join(base_folder, track)))[1]

        dates = []

        for fold in folders:
            if len(fold) == 8:
                if not fold in dates:
                    dates.append(fold)
            if len(fold) == 21:
                if not fold[:8] in dates:
                    dates.append(fold[:8])
                if not fold[9:17] in dates:
                    dates.append(fold[9:17])
        dates = sorted([date for date in dates if (int(date) < 20161011 or int(date) > 20161027)])
        dates = sorted([date for date in dates if (int(date) < 20160113 or int(date) > 20151231)])

        # Download the files
        for date in dates:
            mos_zipfile = os.path.join('/data/mos/meteo/wm/' + date[:4] + '/HARM38', 'HARM38_' + date + analysis_time + '.tgz')
            filename = 'HA38_N25_' + date + forecast_time
            dest_path = os.path.join(grib_folder, track)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            dest = os.path.join(dest_path, filename)

            print(mos_zipfile)
            print(filename)
            print(dest)

            print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

            if not os.path.exists(dest):
                try:
                    tar = tarfile.open(mos_zipfile, 'r')
                    tar.extract(filename, dest)
                except:
                    print('File cannot be found')
                    continue

            # Now remove all not needed variables to reduce filesize
            in_file = os.path.join(dest, filename)
            out_file = dest + '_needed_vars'
            if not os.path.exists(out_file) and os.path.exists(in_file):

                dat = pygrib.open(in_file)

                for g in dat:
                    print(g)

                # Select variables to save..
                out = open(out_file, 'wb')

                for var_name in var_num:
                    var = dat.select(parameterName=var_name)
                    for variable in var:
                        msg = variable.tostring()
                        out.write(msg)