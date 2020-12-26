#!/bin/bash
#SBATCH --job-name=array-job     # create a short name for your job
#SBATCH --output=slurm-%A.%a.out # STDOUT file
#SBATCH --error=slurm-%A.%a.err  # STDERR file
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-7              # job_array
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=changgoo@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

BASEDIR="/tigress/changgoo/TIGRESS-NCR/R8s_8pc_NCR.B2/"
export PYTHONPATH="/home/changgoo/pyathena"

module load anaconda3
conda activate /tigress/changgoo/miniconda3

python /home/changgoo/pyathena/pyathena/tigress_rt/do_tasks.py -b $BASEDIR
