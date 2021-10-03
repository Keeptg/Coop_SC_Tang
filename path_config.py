import os
import sys
import socket

hostname = socket.gethostname()
script_path = sys.path[0]


if hostname in ['login01', 'login02'] or 'node' in hostname:
    root_dir = '/home/users/lifei/Data/Pro_piao'
    data_dir = os.path.join(root_dir, 'Data')
    working_dir = os.environ['WORKDIR']
    cluster_dir = '/home/www/lifei/run_output/Pro_piao'
    model_dir = os.path.join(root_dir, 'model_output')
    for dir_ in [root_dir, data_dir, working_dir, cluster_dir, model_dir]:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
    run_in_cluster = True
elif hostname == 'WGJ-Group':
    root_dir = '/home/lifei/Data/code_project/Pro_piao'
    data_dir = os.path.join(root_dir, 'Data')
    working_dir = os.path.join(root_dir, 'working_dir')
    cluster_dir = os.path.join(root_dir, 'cluster_output')
    model_dir = os.path.join(root_dir, 'model_output')
    for dir_ in [root_dir, data_dir, working_dir, cluster_dir, model_dir]:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
    run_in_cluster = False
else:
    root_dir = os.path.split(sys.path[0])[0]
    data_dir = os.path.join(root_dir, 'Data')
    working_dir = os.path.join(root_dir, 'working_dir')
    cluster_dir = os.path.join(root_dir, 'cluster_output')
    model_dir = os.path.join(root_dir, 'model_output')
    for dir_ in [root_dir, data_dir, working_dir, cluster_dir, model_dir]:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
    run_in_cluster = False
