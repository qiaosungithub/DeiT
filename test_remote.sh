# Run job in a remote TPU VM
source ka.sh # import VM_NAME, ZONE

echo $VM_NAME $ZONE

CONFIG=tpu
CONDA_ENV=DYY

# some of the often modified hyperparametes:
batch=1024
lr=0.001
ep=300
TASKNAME=DeiT

LOGDIR=$(pwd)/tmp
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
python3 -c 'import jax; print(jax.devices())'
" 2>&1 | tee -a $LOGDIR/output.log
# which python3

# python3 -c 'import jax; print(jax.devices())'

# python3 main.py \
#     --workdir=${LOGDIR} --config=configs/${CONFIG}.py \
#     --config.dataset.root='/${DATA_ROOT}/data/imagenet' \
#     --config.batch_size=${batch} \
#     --config.num_epochs=${ep} \
#     --config.learning_rate=${lr} \
#     --config.dataset.prefetch_factor=2 \
#     --config.dataset.num_workers=32 \
#     --config.log_per_step=20 \
#     --config.model='ViT_base'