import os

import jax
import jax.numpy as jnp
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


def _recursive_copy(target, source):
  """Recursively copy values from source into target, preserving target structure.

  Only copies values for keys that exist in target. This handles structure
  mismatches between checkpoints (e.g. missing bias in final_ln, extra _model
  nesting in embedding) without changing the Phase 2 param tree shape.
  """
  for k in list(target.keys()):
    if k not in source:
      continue
    if isinstance(target[k], dict) and isinstance(source[k], dict):
      _recursive_copy(target[k], source[k])
    elif isinstance(target[k], dict) and hasattr(source[k], 'keys'):
      _recursive_copy(target[k], source[k])
    else:
      target[k] = jnp.array(source[k])


def load_backbone_params(state, load_backbone_from, workdir):
  """Load backbone params from a Phase 1 checkpoint into a Phase 2 state.

  Copies all params except 'fc' and 'diffusion_head' from the checkpoint,
  leaving the Phase 2 head randomly initialized. Resets step/opt_state.
  """
  zone = infer_zone_from_workdir(workdir)
  gs_path = convert_to_gs(load_backbone_from, zone)
  raw_ckpt = checkpoints.restore_checkpoint(gs_path, target=None)
  backbone_params = {k: v for k, v in raw_ckpt['params'].items()
                     if k not in ('fc', 'diffusion_head')}
  # Start from the current (Phase 2) params tree (unfreeze to plain dicts).
  current = unfreeze(state.params)
  # Recursively copy checkpoint values, preserving the Phase 2 tree structure.
  # This tolerates minor structure differences (missing bias, extra nesting, etc.)
  _recursive_copy(current, backbone_params)
  new_params = freeze(current)
  # Reset optimizer state to zeros, preserving the existing frozen params tree structure.
  # We cannot call tx.init(new_params) because the WD mask (built from plain dicts in
  # create_train_state) is incompatible with FrozenDict params. Instead, zero the
  # existing opt_state (which has the correct frozen structure) and set step=0.
  new_opt_state = jax.tree_util.tree_map(jnp.zeros_like, state.opt_state)
  return state.replace(params=new_params, opt_state=new_opt_state, step=0)
