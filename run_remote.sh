# Your configurations here

############# COMMON CONFIG #############
CONDA_ENV=DYY # ONLY change this if you are using "eu" machine
TASKNAME=DeiT


############# JOB CONFIG #############
batch=1024
lr=0.00025 # Note that this should be 0.25 times the LR you want (i.e. in the table)
ep=300
CONFIG=tpu
model=ViT_base

############# No need to modify #############
for i in {1..20}; do echo "Do you remember to use TMUX?"; done
source ka.sh

echo Running at $VM_NAME $ZONE

now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
JOBNAME=${TASKNAME}/${now}_${salt}_${VM_NAME}_${CONFIG}_b${batch}_lr${lr}_ep${ep}_torchvision_r50_eval

############# No need to modify [END] #############

############# Change this if you want to reload checkpoint #############
LOGDIR=/$DATA_ROOT/logs/$USER/$JOBNAME
# LOGDIR=/kmh-nfs-ssd-eu-mount/logs/zhh/DeiT/20241007_222941_ysau8g_kmh-tpuvm-v3-32-1_tpu_b1024_lr0.1_ep100_torchvision_r50_eval # reload checkpoint


############# No need to modify #############
sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}
echo 'Log dir: '$LOGDIR

if [[ $USE_CONDA == 1 ]]; then
    CONDA_PATH=$(which conda)
    CONDA_INIT_SH_PATH=$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh
elif [[ $USE_CONDA == 2 ]]; then
    CONDA_INIT_SH_PATH=/kmh-nfs-us-mount/code/zhh/anaconda3/etc/profile.d/conda.sh
    CONDA_ENV=DYY
fi
############# No need to modify [END] #############

################# RUNNING CONFIGS #################

# NOTE: You must use num_workers=64, otherwise, the code will exit unexpectedly

    gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
cd $STAGEDIR
if [ \"$USE_CONDA\" -eq 1 ]; then
    echo 'Using conda'
    source $CONDA_INIT_SH_PATH
    conda activate $CONDA_ENV
fi
echo Current dir: $(pwd)
which python
which pip3
python3 main.py \
    --workdir=${LOGDIR} --config=configs/${CONFIG}.py \
    --config.dataset.root='/${DATA_ROOT}/data/imagenet' \
    --config.batch_size=${batch} \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=64 \
    --config.log_per_step=20 \
    --config.model=${model}
" 2>&1 | tee -a $LOGDIR/output.log
