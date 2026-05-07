import os
import subprocess


DATA_ROOT = "kmh-nfs-ssd-us-mount"
LOCAL_IMAGENET_ROOT = f"/{DATA_ROOT}/data/imagenet"
REMOTE_IMAGENET_ROOT = "/dev/shm/tmp_data/imagenet"


def infer_zone_from_workdir(workdir):
  candidates = [
    "us-central2-b",
    "us-central1-a",
    "us-central1-b",
    "us-east1-d",
    "us-east5-a",
    "us-east5-b",
    "asia-northeast1-b",
    "europe-west4-a",
    "us-central2",
    "us-central1",
    "us-east1",
    "us-east5",
    "europe-west4",
  ]
  matches = [z for z in candidates if z in workdir]
  if not matches:
    return None
  matches.sort(key=len, reverse=True)
  return matches[0]


def _infer_region(zone):
  if zone is None:
    return None
  parts = zone.split("-")
  if len(parts) >= 3 and len(parts[-1]) == 1:
    return "-".join(parts[:-1])
  return zone


def _ensure_imagenet_cache(zone):
  train_dir = os.path.join(REMOTE_IMAGENET_ROOT, "train")
  val_dir = os.path.join(REMOTE_IMAGENET_ROOT, "val")
  if os.path.isdir(train_dir) and os.path.isdir(val_dir):
    return

  os.makedirs("/dev/shm/tmp_data", exist_ok=True)
  gcs_root = os.environ.get("IMAGENET_GCS_ROOT", "")
  if not gcs_root:
    region = _infer_region(zone)
    candidates = []
    if zone:
      candidates.append(f"gs://kmh-gcp-{zone}/data/imagenet/imagenet")
    if region and region != zone:
      candidates.append(f"gs://kmh-gcp-{region}/data/imagenet/imagenet")

    for candidate in candidates:
      check = subprocess.run(
        ["gsutil", "ls", "-d", candidate],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
      )
      if check.returncode == 0:
        gcs_root = candidate
        break

  if not gcs_root:
    raise RuntimeError(f"Failed to resolve ImageNet GCS path for zone: {zone}")

  subprocess.run(["gsutil", "-m", "cp", "-r", gcs_root, "/dev/shm/tmp_data"], check=True)


def resolve_and_prepare_dataset_root(config, workdir):
  if "dataset" not in config or "root" not in config.dataset:
    raise ValueError("Dataset root not found in config")
    return

  root = config.dataset.root
  if not isinstance(root, str):
    raise ValueError("Dataset root is not a string")
    return

  root = os.path.expandvars(root)
  if not root.endswith("/data/imagenet"):
    config.dataset.root = root
    raise ValueError("Dataset root does not end with /data/imagenet")
    return

  zone = infer_zone_from_workdir(workdir)
  if zone == "us-central2-b":
    config.dataset.root = LOCAL_IMAGENET_ROOT
    return

  config.dataset.root = REMOTE_IMAGENET_ROOT
  _ensure_imagenet_cache(zone)
