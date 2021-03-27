import os

run_in_cluster = False
script_dir = os.path.split(os.path.realpath(__file__))[0]
root_dir = os.path.split(script_dir)[0]
data_dir = os.path.join(root_dir, 'Data')
if run_in_cluster:
    working_dir = os.environ['WORKDIR']
else:
    working_dir = os.path.join(root_dir, 'working_dir')

if run_in_cluster:
    cluster_dir = '/home/www/lifei/run_output/Pro_piao'
else:
    cluster_dir = os.path.join(root_dir, 'cluster_output')
model_dir = os.path.join(root_dir, 'model_output')