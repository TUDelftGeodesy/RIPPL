import os, sys
import shutil
import pysftp

def find_paths(stack_folder, output_folder, filename, server=True, host='hpc03.tudelft.net', username='gertmulder', password='', type='image', network=True, convert=True, specific_burst=''):

    stack_folder = os.path.join(stack_folder, 'stack')
    if server:
        sftp = pysftp.Connection(host, username, password=password)

    if server:
        if sftp.isdir(stack_folder):
            remote_dat = sftp.listdir(stack_folder)
            folders = [folder for folder in remote_dat if sftp.isdir(os.path.join(stack_folder, folder))]

    else:
        if os.path.exists(stack_folder):
            folders = next(os.walk(stack_folder))[1]
        else:
            print('source folder does not exist')
            return

    if network:
        folders = [folder for folder in folders if len(folder) == 21]
    else:
        folders = [folder for folder in folders if len(folder) == 8]


    copy_in = []
    copy_out = []

    if type == 'image':
        for folder in folders:
            in_file = os.path.join(stack_folder, folder, filename)
            out_file = os.path.join(output_folder, folder + '_' + filename)
            copy_in.append(in_file)
            copy_out.append(out_file)

    if type == 'image_folder':
        for folder in folders:
            in_file = os.path.join(stack_folder, folder, filename)
            out_file = os.path.join(output_folder, folder, filename)
            copy_in.append(in_file)
            copy_out.append(out_file)

    if type == 'burst':
        for folder in folders:
            if server:
                burst_paths, burstnames = find_burst_folders(os.path.join(stack_folder, folder), server, sftp)
            else:
                burst_paths, burstnames = find_burst_folders(os.path.join(stack_folder, folder))

            for burst, burst_name in zip(burst_paths, burstnames):
                in_file = os.path.join(stack_folder, folder, burst, filename)
                out_file = os.path.join(output_folder, folder + '_' + burst_name + '_' + filename)
                copy_in.append(in_file)
                copy_out.append(out_file)

    if type == 'specific_burst':
        for folder in folders:
            if server:
                burst_paths, burstnames = find_burst_folders(os.path.join(stack_folder, folder), server, sftp)
            else:
                burst_paths, burstnames = find_burst_folders(os.path.join(stack_folder, folder))

            for burst, burst_name in zip(burst_paths, burstnames):
                if burst_name == specific_burst:
                    in_file = os.path.join(stack_folder, folder, burst, filename)
                    out_file = os.path.join(output_folder, folder + '_' + burst_name + '_' + filename)
                    copy_in.append(in_file)
                    copy_out.append(out_file)

    if type == 'seperate_burst':
        for folder in folders:
            if server:
                burst_paths, burstnames = find_burst_folders(folder, server, sftp)
            else:
                burst_paths, burstnames = find_burst_folders(folder)

            for burst, burst_name in zip(burst_paths, burstnames):
                in_file = os.path.join(stack_folder, folder, burst, filename)
                out_file = os.path.join(output_folder, burst, folder + '_' + filename)
                copy_in.append(in_file)
                copy_out.append(out_file)

    for src, dest in zip(copy_in, copy_out):
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        if server:
            if not sftp.isfile(src):
                print(src + ' does not exist')
                continue
        else:
            if not os.path.exists(src):
                print(src + ' does not exist')
                continue

        if not os.path.exists(dest) and (convert == False or (convert == True and not os.path.exists(dest[:-3] + 'jpg'))):
            if server:
                sftp.get(src, dest)
            else:
                shutil.copyfile(src, dest)

            if convert and src.endswith('ras'):
                os.system('convert ' + dest + ' ' + dest[:-3] + 'jpg')
                os.system('rm ' + dest)

def find_burst_folders(folder, sftp_use=False, sftp=''):
    # This function finds all the burst folders in an image folder
    folders = []
    burst_name = []

    if sftp_use:
        remote_dat = sftp.listdir(folder)
        swaths = [fold for fold in remote_dat if sftp.isdir(os.path.join(folder, fold))]
    else:
        swaths = next(os.walk(folder))[1]

    for swath in swaths:
        if sftp_use:
            remote_dat = sftp.listdir(os.path.join(folder, swath))
            bursts = [fold for fold in remote_dat if sftp.isdir(os.path.join(folder, swath, fold))]
        else:
            bursts = next(os.walk(os.path.join(folder, swath)))[1]

        for burst in bursts:
            folders.append(os.path.join(folder, swath, burst))
            burst_name.append(swath + '_' + burst)

    return folders, burst_name


# Actually execute the code...
if __name__ == "__main__":

    type = 'image'
    convert = '1'
    network = '1'

    if len(sys.argv) > 3:
        stack_folder = sys.argv[1]
        output_folder = sys.argv[2]
        filename = sys.argv[3]
    else:
        print('You should at least define the stack_folder, output_folder and filename!')
        sys.exit()
    if len(sys.argv) > 4:
        type = sys.argv[4]
    if len(sys.argv) > 5:
        network = sys.argv[5]
    if len(sys.argv) > 6:
        convert = sys.argv[6]

    find_paths(stack_folder, output_folder, filename, type, convert, network)