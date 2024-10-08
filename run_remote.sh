# Run job in a remote TPU VM
source ka.sh # import VM_NAME, ZONE

echo $VM_NAME $ZONE

CONFIG=tpu

# some of the often modified hyperparametes:
batch=1024
lr=0.001
ep=300
TASKNAME=DeiT

now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
JOBNAME=${TASKNAME}/${now}_${salt}_${VM_NAME}_${CONFIG}_b${batch}_lr${lr}_ep${ep}_torchvision_r50_eval

LOGDIR=/kmh-nfs-ssd-eu-mount/logs/$USER/$JOBNAME
sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}
CONDA_PATH=$(which conda)
CONDA_INIT_SH_PATH=$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh
echo 'Log dir: '$LOGDIR
#  look at bashrc and get conda path

# if USE_CONDA is set, then activate conda environment
if [[ $USE_CONDA -eq 1 ]]; then
    gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
cd $STAGEDIR
source $CONDA_INIT_SH_PATH
conda activate DYY
echo Current dir: $(pwd)
python3 main.py \
    --workdir=${LOGDIR} --config=configs/${CONFIG}.py \
    --config.dataset.root='/${DATA_ROOT}/data/imagenet' \
    --config.batch_size=${batch} \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.log_per_step=20 \
    --config.model='ViT_base'
" 2>&1 | tee -a $LOGDIR/output.log
else
    gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
cd $STAGEDIR
echo Current dir: $(pwd)
python3 main.py \
    --workdir=${LOGDIR} --config=configs/${CONFIG}.py \
    --config.dataset.root='/${DATA_ROOT}/data/imagenet' \
    --config.batch_size=${batch} \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.log_per_step=20 \
    --config.model='ViT_base'
" 2>&1 | tee -a $LOGDIR/output.log
fi

