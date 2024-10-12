source ka.sh # import VM_NAME, ZONE

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
pip3 list | grep matplotlib
echo 牛魔王
"

# which python3
# which pip3
# pip3 list | grep matplotlib
# echo 牛魔王
# pip3 install --upgrade matplotlib
# which python3
# which pip3