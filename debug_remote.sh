# Run job in a remote TPU VM
source ka.sh # import VM_NAME, ZONE

echo $VM_NAME $ZONE

CONFIG=tpu
CONDA_ENV=DYY

# some of the often modified hyperparametes:
batch=32
lr=0.001
ep=10
TASKNAME=DeiT

now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
JOBNAME=${TASKNAME}/${now}_${salt}_${VM_NAME}_${CONFIG}_b${batch}_lr${lr}_ep${ep}_torchvision_r50_eval
STAGEDIR=$(pwd)
LOGDIR=/$DATA_ROOT/logs/$USER/$JOBNAME
# LOGDIR=/kmh-nfs-ssd-eu-mount/logs/zhh/DeiT/20241007_222941_ysau8g_kmh-tpuvm-v3-32-1_tpu_b1024_lr0.1_ep100_torchvision_r50_eval # reload checkpoint
sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}
echo 'Log dir: '$LOGDIR
#  look at bashrc and get conda path

# if USE_CONDA is set, then activate conda environment

if [[ $USE_CONDA == 1 ]]; then
    CONDA_PATH=$(which conda)
    CONDA_INIT_SH_PATH=$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh
elif [[ $USE_CONDA == 2 ]]; then
    CONDA_INIT_SH_PATH=/kmh-nfs-us-mount/code/zhh/anaconda3/etc/profile.d/conda.sh
    CONDA_ENV=DYY
fi

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
cd $STAGEDIR
source $CONDA_INIT_SH_PATH
conda activate $CONDA_ENV
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
    --config.model='ViT_debug'
" 2>&1 | tee -a $LOGDIR/output.log

