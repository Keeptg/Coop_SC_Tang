#!/bin/bash
#
#SBATCH --job-name=run_climate1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --mail-user=lifei@itpcas.ac.cn
#SBATCH --mail-type=ALL
#SBATCH --array=0-1

# Abort whenever a single step fails. Without this, bash will just continue on errors.
set -e

# On every node, when slurm starts a job, it will make sure the directory
# /work/username exists and is writable by the jobs user.
# We create a sub-directory there for this job to store its runtime data at.
WORKDIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/my_run"
JOBID=$SLURM_ARRAY_JOB_ID
TASK_ID=$SLURM_ARRAY_TASK_ID
mkdir -p "$WORKDIR"
echo "Workdir for this run: $WORKDIR"

# Export the WORKDIR as environment variable so our script can use it to find its working directory.
export WORKDIR TASK_ID

# Use the local data download cache
export OGGM_DOWNLOAD_CACHE=/home/data/download
export OGGM_DOWNLOAD_CACHE_RO=1
export OGGM_EXTRACT_DIR="/work/$SLURM_JOB_USER/$SLURM_JOB_ID/oggm_tmp"

# All commands in the EOF block run inside of the container
# Adjust container version to your needs, they are guaranteed to never change after their respective day has passed.
srun -n 1 -c "${SLURM_JOB_CPUS_PER_NODE}" singularity exec /home/users/lifei/images/oggm_20210921.sif bash -s <<EOF
  set -e
  # Setup a fake home dir inside of our workdir, so we don't clutter the actual shared homedir with potentially incompatible stuff.
  export HOME="$WORKDIR/fake_home"
  mkdir "\$HOME"
  # Create a venv that _does_ use system-site-packages, since everything is already installed on the container.
  # We cannot work on the container itself, as the base system is immutable.
  python3 -m venv --system-site-packages "$WORKDIR/oggm_env"
  source "$WORKDIR/oggm_env/bin/activate"
  # Make sure latest pip is installed
  pip install --upgrade pip setuptools
  # OPTIONAL: install another OGGM version
  pip install --upgrade "git+https://github.com/OGGM/oggm.git"
  # pip install --upgrade "git+https://github.com/OGGM/oggm.git@2fbe179521a9ef97e4d6dbcec0c105dbbf279bc3"
  #pip install --upgrade "git+https://github.com/OGGM/oggm.git@af0ac438be77a0f86ae2e33538c894c8a34061db"
  # Finally, the run
  python3 ./run_glc_model_climate1.py

EOF
#echo "Copying files..."
#cp -R "$WORKDIR" /home/users/lifei/my_run_output/ 

# Print a final message so you can actually see it being done in the output log.
echo "SLURM DONE"
