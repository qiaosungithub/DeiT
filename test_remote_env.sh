# Run job in a remote TPU VM
source ka.sh # import VM_NAME, ZONE

echo $VM_NAME $ZONE


gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
which python
which pip3
python3 -c 'import jax; print(jax.devices())'
" 2>&1 | tee -a $LOGDIR/output.log