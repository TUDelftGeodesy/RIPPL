# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:54:00 2012

@author: Piers Titus van der Torren <pierstitus@gmail.com>
Updated: 11-07-2013, Freek van Leijen, TU Delft
"""

#radardbdir = '/export/disk1/home1/everybody/radardb/rs_data/eurasia/nl/'
#zipdir = '/export/disk1/home1/fjvanleijen/new_data/nl_radarsat2_nso'
#radardbdir = '/home/everybody/radardb/rs_data/eurasia/nl/'
radardbdir = '/home/everybody/radardb/radar_data/eurasia/netherlands/rsat2/'
zipdir = '/home/fjvanleijen/new_data/nl_radarsat2_nso'

mail_error = ['F.J.vanLeijen@tudelft.nl']
#mail_notification = ['F.J.vanLeijen@tudelft.nl',
#                     'P.Dheenathayalan@tudelft.nl',
#                     'P.S.Mahapatra@tudelft.nl',
#                     'S.SamieiEsfahany@tudelft.nl',
#                     'H.vanderMarel@tudelft.nl']
mail_notification = ['F.J.vanLeijen@tudelft.nl']

import os
from lxml import etree
from glob import glob
from os import path
from datetime import datetime
from zipfile import ZipFile
from math import sqrt

# Import smtplib for the email sending function
import smtplib
# Import the email modules we'll need
from email.mime.text import MIMEText

import merge_radarsat

def xmltext(f,addr,ns='{http://www.rsi.ca/rs2/prod/xml/schemas}'):
    """Helper to use a default namespace in find/xpath."""
    return f.find(addr.replace('/','/'+ns)).text

def get_metadata(fi):
    """Extract meta_data from a RSAT2 xml file and return as a dict."""
    dateformat = '%Y-%m-%dT%H:%M:%S.%fZ'
    f = etree.parse(fi)
    return {
        'prf': float(xmltext(f,'/sourceAttributes/radarParameters/pulseRepetitionFrequency')),
        't1': datetime.strptime(xmltext(f,'/imageGenerationParameters/sarProcessingInformation/zeroDopplerTimeFirstLine'),dateformat),
        'lines': int(xmltext(f,'/imageAttributes/rasterAttributes/numberOfLines')),
        'pixels': int(xmltext(f,'/imageAttributes/rasterAttributes/numberOfSamplesPerLine')),
        'latitude': float(xmltext(f,'/imageAttributes/geographicInformation/geolocationGrid/imageTiePoint/geodeticCoordinate/latitude')),
        'longitude': float(xmltext(f,'/imageAttributes/geographicInformation/geolocationGrid/imageTiePoint/geodeticCoordinate/longitude')),
        'pass': xmltext(f,'/sourceAttributes/orbitAndAttitude/orbitInformation/passDirection'),
    }

def get_nearest_frame(done, m):
    """Find the existing frame nearest to the given image.

    Args:
      done: list of dicts of existing frames.
      m: meta_data dict of image.
    
    Returns:
      frame, dist: found frame name and distance from given image.
    """
    frame = ''
    dist = 360
    for do in done:
        d = sqrt((do['latitude']-m['latitude'])**2 + (do['longitude']-m['longitude'])**2)
        if d < dist and do['pass'] == m['pass']:
            dist = d
            frame = do['frame']
    return frame, dist

print('getting existing frame info')
files = glob(path.join(radardbdir,'*/data/RS2_OK*/product.xml'))

done = []
for fi in files:
    m = get_metadata(fi)
    m['frame'] = fi.split(path.sep)[-4]
    m['orgname'] = fi.split(path.sep)[-2]
    done.append(m)

done_orgnames = [m['orgname'] for m in done]
done_ident = [m[30:] for m in done_orgnames]
# special case a few frames which are just a second different than exist
ignore = ['S3_20120312_055338_HH_HV_SLC','S3_20120305_055737_HH_HV_SLC']
done_ident.extend(ignore)

# temporary piece to find double frames
#done.sort(key=lambda a:a['orgname'],reverse=True)
#dd={}
#for d in done:
#    id = d['frame'] + d['orgname'][30:40]
#    if id in dd:
#        print(d['frame'] + '/data/' + d['orgname'] + '          ' + {True:' ',False:'*'}[d['orgname'][:20]==dd[id]['orgname'][:20]] + '\t' + dd[id]['orgname'])
#    else:
#        dd[id] = d

#'RS2_OK24779_PK282242_DK251667_S3_20120312_055338_HH_HV_SLC'
print('processing new files')

succeeded = []
errors = []
for fi in glob(path.join(zipdir,'RS2_*_SLC.[Zz][Ii][Pp]')):
    try:
        orgname = path.basename(fi)[:-4]
        ident = orgname[30:]
        if not ident in done_ident:
            m = get_metadata(ZipFile(fi).open(path.join(orgname, 'product.xml')))
            frame, d = get_nearest_frame(done, m)
            print('{0} is in frame {1}, dist {2}'.format(orgname, frame, d))
            if d < 0.1:
                # distance to nearest frame must be < 10km (~0.1 degree) to automatically unzip
                print('extracting zip')
                ZipFile(fi).extractall(path.join(radardbdir, frame, 'data'))
                note = '{0} in frame {1}'.format(orgname, frame)
                succeeded.append(note)
            else:
                print('Distance to existing frames too large, not extracting {0}. Please do manually'.format(orgname))
                note = 'No frame match for {0}. Please extract manually'.format(orgname)
                print(note)
                errors.append(note)
    except Exception as e:
        print(str(e.__class__) + ': ' + str(e))
        print('Not succeeded, continuing with next file')
        print('Unzip error: ' + str(e.__class__) + ': ' + str(e) + '\n' + 'while processing file ' + fi)
        note = 'Unzip error: ' + str(e.__class__) + ': ' + str(e) + '\n' + 'while processing file ' + fi
        errors.append(note)

if errors and mail_error:
    text = 'One or more errors were encountered while updating the RADARSAT2 archive:\n\n' + '\n'.join(errors)
    footer = '\n\nThis message is automatically generated by update_radarsat.py'
    msg = MIMEText(text + footer)
    msg['Subject'] = 'Errors while updating RADARSAT2 archive'
    me = 'F.J.vanLeijen@tudelft.nl'
    msg['From'] = '"RADARSAT2 archive updater" <' + me + '>'
    msg['To'] = 'undisclosed-recipients'

    s = smtplib.SMTP('localhost')
    s.sendmail(msg['From'], mail_error, msg.as_string())
    s.quit()

if succeeded and mail_notification:
    text = 'The RADARSAT2 archive has been updated with the following images:\n\n' + '\n'.join(succeeded)
    footer = '\n\nThis message is automatically generated by update_radarsat.py'
    msg = MIMEText(text + footer)
    msg['Subject'] = 'RADARSAT2 archive update'
    me = 'F.J.vanLeijen@tudelft.nl'
    msg['From'] = '"RADARSAT2 archive updater" <' + me + '>'
    msg['To'] = 'undisclosed-recipients'#','.join(mail_notification)

    s = smtplib.SMTP('smtp.tudelft.nl')
    s.sendmail(me, mail_notification, msg.as_string())
    s.quit()

#if succeeded:
print('merge, coregister and resample new images')
  #merge_radarsat.merge_all()
    #merge_radarsat()
merge_radarsat.main()
    #os.system('nice -n10 ionice -c2 -n5 python /export/disk1/home1/fjvanleijen/stacks/nl_rsat2/nl_west_rsat2_dsc_t102/dorisstack_west_rsat2.py')
    #os.system('nice -n10 ionice -c2 -n5 python /export/disk1/home1/fjvanleijen/stacks/nl_rsat2/nl_center_rsat2_dsc_t202/dorisstack_center_rsat2.py')
    #os.system('nice -n10 ionice -c2 -n5 python /export/disk1/home1/fjvanleijen/stacks/nl_rsat2/nl_eastsouth_rsat2_dsc_t302/dorisstack_eastsouth_rsat2.py')
    #os.system('nice -n10 ionice -c2 -n5 python /export/disk1/home1/fjvanleijen/stacks/nl_rsat2/nl_groningen_rsat2_dsc_t302/dorisstack_groningen_rsat2.py')

