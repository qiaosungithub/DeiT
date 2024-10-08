# Your configurations here

############# COMMON CONFIG #############
CONDA_ENV=DYY # ONLY change this if you are using "eu" machine



############# JOB CONFIG #############
batch=32
lr=0.001
ep=10
CONFIG=tpu
model=ViT_debug

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
    --config.dataset.root='/kmh-nfs-ssd-eu-mount/code/hanhong/data/imagenet_fake/$DATA_ROOT/' \
    --config.batch_size=${batch} \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=64 \
    --config.log_per_step=20 \
    --config.model=${model}
" 2>&1 | tee -a $LOGDIR/output.log