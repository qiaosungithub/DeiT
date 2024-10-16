# Stage code and run job in a remote TPU VM

# ------------------------------------------------
# Copy all code files to staging
# ------------------------------------------------
now=`date '+%y%m%d%H%M%S'`
salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
commitid=`git show -s --format=%h`  # latest commit id; may not be exactly the same as the commit
export STAGEDIR=/kmh-nfs-ssd-eu-mount/staging/$USER/${now}-${salt}-${commitid}-code

echo 'Staging files...'
rsync -a . $STAGEDIR --exclude=tmp --exclude=.git --exclude=__pycache__ --exclude='*.png'
cp -r /kmh-nfs-ssd-eu-mount/code/hanhong/MyFile/research_utils/Jax/zhh $STAGEDIR
echo 'Done staging.'

chmod 777 $STAGEDIR

cd $STAGEDIR
echo 'Current dir: '`pwd`
# ------------------------------------------------

source run_remote.sh