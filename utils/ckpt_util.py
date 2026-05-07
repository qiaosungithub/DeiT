import os

from flax.training import checkpoints


def infer_zone_from_workdir(workdir):
  candidates = [
    'us-central1',
    'us-east1',
    'us-east5',
    'us-central2',
    'asia-northeast1-b',
    'europe-west4',
  ]
  matches = [z for z in candidates if z in workdir]
  if not matches:
    raise ValueError(f'Cannot infer zone from workdir: {workdir}')
  if len(matches) != 1:
    raise ValueError(f'Multiple matched zones {matches} from workdir {workdir}')
  return matches[0]


def convert_to_gs_by_zone(path, zone):
  if zone == 'us-central1':
    return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-central1/qiao_zhicheng_hanhong_files')
  if zone == 'us-east1':
    return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-east1/qiao_zhicheng_hanhong_files')
  if zone == 'us-east5':
    return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-east5/qiao_zhicheng_hanhong_files')
  if zone == 'us-central2':
    return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-us-central2/qiao_zhicheng_hanhong_files')
  if zone == 'asia-northeast1-b':
    return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp-asia-northeast1-b/qiao_zhicheng_hanhong_files')
  if zone == 'europe-west4':
    return path.replace('/kmh-nfs-ssd-us-mount/logs/sqa', 'gs://kmh-gcp/qiao_zhicheng_hanhong_files')
  raise ValueError(f'Unsupported zone {zone}')


def convert_to_gs(path, zone):
  if path.startswith('gs://'):
    return path
  if not os.path.isabs(path):
    raise ValueError(f'Checkpoint path must be absolute: {path}')
  return convert_to_gs_by_zone(path, zone)


def restore_checkpoint(state, load_from, workdir):
  zone = infer_zone_from_workdir(workdir)
  gs_path = convert_to_gs(load_from, zone)
  return checkpoints.restore_checkpoint(gs_path, state)


def save_checkpoint(state, workdir, keep=2):
  zone = infer_zone_from_workdir(workdir)
  gs_workdir = convert_to_gs(workdir, zone)
  return checkpoints.save_checkpoint_multiprocess(gs_workdir, state, int(state.step), keep=keep)
