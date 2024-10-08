rm -rf tmp # Comment this line if you want to reload (usually not the case)

CONDA_PATH=$(which conda)
CONDA_INIT_SH_PATH=$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh
LOGDIR=$(pwd)/tmp
sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}

# hyperparameters
batch=1024
lr=0.0005
ep=100
wd=0.05
grad_norm_clip=None

CONFIG=fake_data_benchmark
source $CONDA_INIT_SH_PATH
# set the device to cpu
export JAX_PLATFORMS=cpu
# remember to use your own conda environment
# conda activate DYY
conda activate wgt

echo "start running main"

python3 main.py \
    --workdir=${LOGDIR} --config=configs/${CONFIG}.py \
    --config.dataset.root="$(pwd)/imagenet_fake" \
    --config.batch_size=${batch} \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.log_per_step=1000 \
    --config.model='ViT_debug' \
    --config.optimizer='adamw' \
    --config.weight_decay=${wd} \
    --config.grad_norm_clip=${grad_norm_clip} \
    --config.debug=True