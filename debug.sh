CONDA_PATH=$(which conda)
CONDA_INIT_SH_PATH=$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh
LOGDIR=$(pwd)/tmp
sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}
batch=1024
lr=0.1
ep=100
CONFIG=fake_data_benchmark
source $CONDA_INIT_SH_PATH
export JAX_PLATFORMS=cpu
# remember to use your own conda environment
# conda activate DYY
conda activate wgt

echo "start running main"

python3 main.py \
    --workdir=${LOGDIR} --config=configs/${CONFIG}.py \
    --config.dataset.root='./imagenet_fake' \
    --config.batch_size=${batch} \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.log_per_step=20 \
    --config.model='ViT_debug'