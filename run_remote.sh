# Your configurations here
source config.sh
CONDA_ENV=$OWN_CONDA_ENV_NAME
############# COMMON CONFIG #############
TASKNAME=DeiT


############# JOB CONFIG #############
batch=1024
lr=0.0005
ep=110 # with repeat-aug, 100 epoch is equivalent to 300 epoch previously # add 10 more epochs for colling down
CONFIG=tpu
model=ViT_base

# other hyperparameters

wd=0.05
label_smoothing=0.1
grad_norm_clip=None

use_rand_augment=1
# if use rand augment:
rand_augment=rand-m9-mstd0.5-inc1
# if use random erasing:
reprob=0.25

use_mixup_cutmix=1
# if use mixup / cutmix:
mixup_prob=1.0
switch_prob=0.5
mixup_alpha=0.8
cutmix_alpha=1.0
repeated_aug=3
dropout_rate=0.0
stochastic_depth_rate=0.1
num_tpus=32

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
echo 'Current dir: '
pwd
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
    --config.model='${model}' \
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
    --config.dataset.num_tpus=${num_tpus} \
" 2>&1 | tee -a $LOGDIR/output.log