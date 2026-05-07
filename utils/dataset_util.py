import os
import subprocess
import time

from utils.ckpt_util import infer_zone_from_workdir


LOCAL_IMAGENET_ROOT = f"/kmh-nfs-ssd-us-mount/data/imagenet"
REMOTE_IMAGENET_ROOT = "/mnt/zhhm/zhh/imagenet/imagenet"
MIN_TRAIN_FILES = 1280000
MIN_VAL_FILES = 50000


def _count_files(path):
  total = 0
  for _, _, files in os.walk(path):
    total += len(files)
  return total


def _imagenet_ready(root):
  train_dir = os.path.join(root, "train")
  val_dir = os.path.join(root, "val")
  if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
    return False
  train_count = _count_files(train_dir)
  val_count = _count_files(val_dir)
  return train_count > MIN_TRAIN_FILES and val_count >= MIN_VAL_FILES


def _prepare_download_env():
  mount_sh = os.path.join(os.path.dirname(__file__), "mount_disk.sh")
  subprocess.run(["sudo", "bash", mount_sh], check=True)


def _infer_region(zone):
  if zone is None:
    return None
  parts = zone.split("-")
  if len(parts) >= 3 and len(parts[-1]) == 1:
    return "-".join(parts[:-1])
  return zone


def _ensure_imagenet_cache(zone):
  if _imagenet_ready(REMOTE_IMAGENET_ROOT):
    return

  _prepare_download_env()

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
        ["gsutil", "ls", f"{candidate}.tar.*"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
      )
      if check.returncode == 0:
        gcs_root = candidate
        break

  if not gcs_root:
    raise RuntimeError(f"Failed to resolve ImageNet GCS path for zone: {zone}")

  files = subprocess.check_output(
    f"gsutil ls {gcs_root}.tar.*",
    shell=True,
  ).decode("utf-8").strip().split("\n")
  files = sorted([f for f in files if f])
  if not files:
    raise RuntimeError(f"No tar parts found at {gcs_root}.tar.*")

  copy_cmd = ["gsutil", "-m", "cp", *files, "/dev/shm/tmp_data"]
  last_err = None
  for i in range(5):
    proc = subprocess.run(copy_cmd, capture_output=True, text=True)
    if proc.returncode == 0:
      last_err = None
      break
    last_err = RuntimeError(
      f"gsutil multipart copy failed (attempt {i + 1}/5):\n"
      f"stdout:\n{proc.stdout}\n"
      f"stderr:\n{proc.stderr}"
    )
    time.sleep(2 * (i + 1))
  if last_err is not None:
    for src in files:
      proc = subprocess.run(
        ["gsutil", "cp", src, "/dev/shm/tmp_data"],
        capture_output=True,
        text=True,
      )
      if proc.returncode != 0:
        raise RuntimeError(
          f"gsutil single-file copy failed for {src}:\n"
          f"stdout:\n{proc.stdout}\n"
          f"stderr:\n{proc.stderr}"
        )

  extract_cmd = (
    "set -eu; "
    "rm -f /dev/shm/zhh_stream; "
    "mkfifo /dev/shm/zhh_stream; "
    "(for f in /dev/shm/tmp_data/*; do "
    "if [ -f \"$f\" ]; then cat \"$f\"; rm -f \"$f\"; fi; "
    "done > /dev/shm/zhh_stream) & "
    "sudo rm -rf /mnt/zhhm/zhh/imagenet; "
    "sudo mkdir -p /mnt/zhhm/zhh/imagenet; "
    "sudo chmod a+r /mnt/zhhm/zhh/imagenet; "
    "sudo tar -C /mnt/zhhm/zhh/imagenet -xf /dev/shm/zhh_stream; "
    "sudo rm -f /dev/shm/zhh_stream; "
    "sudo rm -rf /dev/shm/tmp_data"
  )
  subprocess.run(extract_cmd, shell=True, check=True)

  if not _imagenet_ready(REMOTE_IMAGENET_ROOT):
    raise RuntimeError(
      f"ImageNet extraction failed or incomplete under {REMOTE_IMAGENET_ROOT}"
    )


def resolve_and_prepare_dataset_root(config, workdir):
  if "dataset" not in config or "root" not in config.dataset:
    return

  root = config.dataset.root
  if not isinstance(root, str):
    return

  root = os.path.expandvars(root)
  if not root.endswith("/data/imagenet"):
    config.dataset.root = root
    return

  zone = infer_zone_from_workdir(workdir)
  if zone == "us-central2":
    config.dataset.root = LOCAL_IMAGENET_ROOT
    return

  config.dataset.root = REMOTE_IMAGENET_ROOT
  _ensure_imagenet_cache(zone)
