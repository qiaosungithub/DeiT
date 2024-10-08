# 卡.sh

export VM_NAME=kmh-tpuvm-v3-32-1
export ZONE=europe-west4-a
# export ZONE=us-central1-a

# if 'europe' in ZONE, then set type to "eu"
# otherwise, set type to "us"
if [[ $ZONE == *"europe"* ]]; then
    export DATA_ROOT="kmh-nfs-ssd-eu-mount"
    export USE_CONDA=1
else
    export DATA_ROOT="kmh-nfs-us-mount"
    export USE_CONDA=2
fi