# Your configurations here
source config.sh
CONDA_ENV=$OWN_CONDA_ENV_NAME

############# JOB CONFIG #############
batch=32
lr=0.001
ep=10
CONFIG=tpu
model=ViT_debug

# sqa

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
switch_prob=0.5
mixup_alpha=0.8
cutmix_alpha=1.0
repeated_aug=3
dropout_rate=0.0
stochastic_depth_rate=0.1

############# No need to modify #############
source ka.sh

echo Running at $VM_NAME $ZONE

STAGEDIR=/$DATA_ROOT/staging/$(whoami)/debug
mkdir -p $STAGEDIR
sudo chmod 777 $STAGEDIR
echo 'Staging files...'
rsync -a . $STAGEDIR --exclude=tmp --exclude=.git --exclude=__pycache__
echo 'Done staging.'

LOGDIR=$STAGEDIR/log
rm -rf $LOGDIR
sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}
echo 'Log dir: '$LOGDIR


if [[ $USE_CONDA == 1 ]]; then
    CONDA_PATH=$(which conda)
    CONDA_INIT_SH_PATH=$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh
fi
############# No need to modify [END] #############


################# RUNNING CONFIGS #################
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
    --config.dataset.root=${FAKE_DATA_ROOT} \
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
" 2>&1 | tee -a $LOGDIR/output.log