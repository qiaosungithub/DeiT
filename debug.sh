source config.sh
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
label_smoothing=0.1
grad_norm_clip=None

use_rand_augment=1
# if use rand augment:
rand_augment=rand-m9-mstd0.5-inc1
# if use random erasing:
reprob=0.25

use_mixup_cutmix=1
# if use mixup / cutmix: (i currently do not know what is the paper's parameter)
mixup_prob=1.0
mixup_alpha=0.8
cutmix_alpha=1.0
switch_prob=0.5
repeated_aug=3
dropout_rate=0.0
stochastic_depth_rate=0.1

CONFIG=fake_data_benchmark
# CONFIG=tpu
source $CONDA_INIT_SH_PATH
export JAX_PLATFORMS=cpu
# remember to use your own conda environment
conda activate $OWN_CONDA_ENV_NAME

echo "start running main"

python3 main.py \
    --workdir=${LOGDIR} --config=configs/${CONFIG}.py \
    --config.dataset.root=${EU_IMAGENET_FAKE} \
    --config.debug=True \
    --config.batch_size=${batch} \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=64 \
    --config.log_per_step=20 \
    --config.model='ViT_debug' \
    --config.optimizer='adamw' \
    --config.weight_decay=${wd} \
    --config.grad_norm_clip=${grad_norm_clip} \
    --config.dropout_rate=${dropout_rate} \
    --config.stochastic_depth_rate=${stochastic_depth_rate} \
    --config.dataset.label_smoothing=${label_smoothing} \
    --config.dataset.use_rand_augment=${use_rand_augment} \
    --config.dataset.rand_augment=${rand_augment} \
    --config.dataset.reprob=${reprob} \
    --config.dataset.use_mixup_cutmix=${use_mixup_cutmix} \
    --config.dataset.mixup_prob=${mixup_prob} \
    --config.dataset.mixup_alpha=${mixup_alpha} \
    --config.dataset.cutmix_alpha=${cutmix_alpha} \
    --config.dataset.switch_prob=${switch_prob} \
    --config.dataset.repeated_aug=${repeated_aug} \