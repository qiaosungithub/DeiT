# initialize and set up remote TPU VM
source ka.sh # import VM_NAME, ZONE

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
pip3 install --upgrade timm

pip3 install absl-py==1.4.0
pip3 install clu==0.0.11
pip3 install flax==0.8.1
pip3 install jax[tpu]==0.4.25 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install ml-collections==0.1.1
pip3 install numpy==1.26.4
pip3 install optax==0.2.1
pip3 install tensorflow==2.15.0.post1
pip3 install torch==2.2.2
pip3 install torchvision==0.17.2
pip3 install orbax-checkpoint==0.4.4
pip3 install chex==0.1.86
pip3 install matpotlib==3.9.2
pip3 install jax[tpu]==0.4.25 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

"

echo $VM_NAME $ZONE

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
which python3
python3 -c 'import jax; print(jax.device_count())'
"

python3 -c 'import jax; print(jax.device_count())'