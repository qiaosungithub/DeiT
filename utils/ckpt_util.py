import os

import jax
from flax.training import checkpoints
from flax.core import freeze, unfreeze


def infer_zone_from_workdir(workdir):
  candidates = [
    'us-central1',
    'us-east1',
    'us-east5',
    'us-central2',
    'asia-northeast1-b',
    'europe-west4',
    'code/qiao/work',
  ]
  matches = [z for z in candidates if z in workdir]
  if not matches:
    raise ValueError(f'Cannot infer zone from workdir: {workdir}')
  if len(matches) != 1:
    raise ValueError(f'Multiple matched zones {matches} from workdir {workdir}')
  if matches[0] == 'code/qiao/work':
    return 'us-central2'
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


def load_backbone_params(state, load_backbone_from, workdir):
  """Load backbone params from a Phase 1 checkpoint into a Phase 2 state.

  Copies all params except 'fc' and 'diffusion_head' from the checkpoint,
  leaving the Phase 2 head randomly initialized. Resets step/opt_state.
  """
  import jax.numpy as jnp
  zone = infer_zone_from_workdir(workdir)
  gs_path = convert_to_gs(load_backbone_from, zone)
  raw_ckpt = checkpoints.restore_checkpoint(gs_path, target=None)
  backbone_params = {k: v for k, v in raw_ckpt['params'].items()
                     if k not in ('fc', 'diffusion_head')}
  # Start from the current (Phase 2) params tree (unfreeze to plain dicts).
  current = unfreeze(state.params)
  # Copy backbone values, converting numpy arrays to JAX arrays to ensure
  # consistent pytree node types throughout.
  for key, val in backbone_params.items():
    current[key] = jax.tree_util.tree_map(jnp.array, val)
  new_params = freeze(current)
  # Reinit optimizer state. optax.masked expects plain dicts (not FrozenDict),
  # so unfreeze before init then re-store frozen params.
  new_opt_state = state.tx.init(unfreeze(new_params))
  return state.replace(params=new_params, opt_state=new_opt_state, step=0)
