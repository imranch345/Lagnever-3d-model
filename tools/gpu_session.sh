#!/bin/bash
# Level 2b GPU sessions on Google Cloud: one L4 VM with the safeguards in docs/LEVEL_2B_SPEND.md.
#
#   tools/gpu_session.sh launch RUNS_FILE DATA_DIR   create the VM, upload, resume or start runs
#   tools/gpu_session.sh deploy RUNS_FILE DATA_DIR   the same onto the VM that already exists
#   tools/gpu_session.sh pull                       copy every run's files back to the Mac
#   tools/gpu_session.sh status                     VM state and the last line of each run log
#   tools/gpu_session.sh delete                     delete the VM
#   tools/gpu_session.sh start                      restart a VM the watchdog stopped (new IP)
#   tools/gpu_session.sh stage DATA_DIR             keep DATA_DIR in the project as a snapshot
#
# RUNS_FILE has one run per line, "name|fields.train arguments" (tools/gpu_runs/*.txt).
# DATA_DIR is a fields.data output folder (manifest.json and points/).
# PROVISIONING=SPOT (default) or STANDARD (on-demand, never reclaimed).
#
# Safeguards: the VM deletes itself after 6 hours; it has no service account and no API
# scopes; SSH uses a dedicated key in instance metadata with project keys blocked. A run
# resumes from the newest state.pt on the Mac, so losing the VM costs one save interval.
# If deploying fails, launch deletes the VM. On the VM, a watchdog (startup script) shuts it
# down after 45 minutes with no training or upload, so an idle VM bills only its disk.
# Uploads use rsync --partial with retries, so a dropped connection resumes the transfer.
#
# Data. Uploading 430 MB from the Mac takes 25-50 minutes, too long to pay at GPU prices on
# every VM. `stage` uploads it once, to a small CPU VM, and keeps it as the disk snapshot
# lagnav-DATA_NAME (about $0.03 a month). When that snapshot exists, `launch` gives the GPU VM
# a data disk made from it and uploads only the code and the runs' resume states.
set -euo pipefail

PROJECT=lagnever-ai-app-n
NAME=lagnav-2b-m2
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="$REPO/datasets/processed/level2b/runs/gpu"
STATE="$HOME/.lagnav/gpu_state"
KEY="$HOME/.lagnav/gcp_lagnav_ed25519"
KNOWN="$HOME/.lagnav/gcp_known_hosts"
CACHE="$HOME/.lagnav/cache"
ZONES="us-east1-b us-east1-c us-east1-d us-central1-a us-central1-b us-central1-c us-west1-a us-west1-b us-west1-c asia-south1-a asia-south1-b asia-south1-c"

vm() {  # run a command on the VM; -n so it never reads this script's stdin
  # shellcheck disable=SC1090
  source "$STATE"
  ssh -n -i "$KEY" -o UserKnownHostsFile="$KNOWN" -o StrictHostKeyChecking=accept-new \
    -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 "lagnav@${IP}" "$@"
}

upload() {  # upload DEST FILE...: DEST relative to the VM's home; resumes a dropped transfer
  local dest="$1"
  shift
  # shellcheck disable=SC1090
  source "$STATE"
  for attempt in 1 2 3 4 5; do
    if rsync -a --partial --timeout=120 \
        -e "ssh -i $KEY -o UserKnownHostsFile=$KNOWN -o BatchMode=yes -o ConnectTimeout=15" \
        "$@" "lagnav@${IP}:$dest"; then
      return 0
    fi
    echo "upload attempt $attempt failed; retrying"
    sleep 20
  done
  return 1
}

data_tar() {  # the cached tar of DATA_DIR's manifest and points, rebuilt when they change
  local data_dir="$1" data_name
  data_name="$(basename "$data_dir")"
  mkdir -p "$CACHE"
  if [ ! -f "$CACHE/$data_name.tar" ] || [ "$data_dir/manifest.json" -nt "$CACHE/$data_name.tar" ]; then
    COPYFILE_DISABLE=1 tar --no-xattrs -cf "$CACHE/$data_name.tar" \
      -C "$(dirname "$data_dir")" "$data_name/manifest.json" "$data_name/points"
  fi
  echo "$CACHE/$data_name.tar"
}

create() {  # create [SNAPSHOT]: the GPU VM, with a data disk made from SNAPSHOT if given
  local provisioning="${PROVISIONING:-SPOT}" keys watchdog
  local -a data_disk=()
  if [ -n "${1:-}" ]; then
    data_disk=(--create-disk "name=$NAME-data,source-snapshot=$1,device-name=lagnav-data,type=pd-balanced,auto-delete=yes")
  fi
  keys="$(mktemp)"
  printf 'lagnav:%s\n' "$(cat "$KEY.pub")" > "$keys"
  watchdog="$(mktemp)"
  cat > "$watchdog" <<'EOF'
#!/bin/bash
# Shut down after 45 minutes with neither training nor an upload running: a stopped VM
# bills only its disk. (rsync --server is the receiving end of an upload.)
idle=0
while true; do
  sleep 60
  if pgrep -f 'fields.train|rsync --server' > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  if [ "$idle" -ge 45 ]; then shutdown -h now; fi
done
EOF
  for zone in $ZONES; do
    if gcloud compute instances create "$NAME" --project "$PROJECT" --zone "$zone" \
        --machine-type g2-standard-8 --provisioning-model "$provisioning" \
        --instance-termination-action DELETE --max-run-duration 6h \
        --maintenance-policy TERMINATE \
        --image-family pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
        --image-project deeplearning-platform-release \
        --boot-disk-size 100GB --boot-disk-type pd-balanced ${data_disk[@]+"${data_disk[@]}"} \
        --no-service-account --no-scopes \
        --metadata-from-file "ssh-keys=$keys,startup-script=$watchdog" \
        --metadata block-project-ssh-keys=TRUE,install-nvidia-driver=True \
        --labels purpose=lagnav-level2b,milestone=m2 > /dev/null 2>&1; then
      local ip
      ip="$(gcloud compute instances describe "$NAME" --project "$PROJECT" --zone "$zone" \
        --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
      printf 'ZONE=%s\nIP=%s\n' "$zone" "$ip" > "$STATE"
      # A new VM can reuse a released IP with a new host key; forget the old one.
      ssh-keygen -R "$ip" -f "$KNOWN" > /dev/null 2>&1 || true
      rm -f "$keys" "$watchdog"
      echo "created $provisioning in $zone at $(date -u +%H:%M:%S) UTC"
      return 0
    fi
    echo "no $provisioning capacity in $zone"
  done
  rm -f "$keys" "$watchdog"
  echo "no zone had $provisioning capacity" >&2
  return 1
}

launch() {
  local snapshot
  snapshot="lagnav-$(basename "$2")"
  if gcloud compute snapshots describe "$snapshot" --project "$PROJECT" > /dev/null 2>&1; then
    echo "data from snapshot $snapshot"
  else
    snapshot=""
  fi
  create "$snapshot"
  # A separate process, so that set -e still stops it at the first failure.
  if ! "$0" deploy "$1" "$2"; then
    echo "deploy failed; deleting the VM so it does not sit idle" >&2
    delete_vm
    return 1
  fi
}

deploy() {
  local runs_file="$1" data_dir="$2" data_name stage
  data_name="$(basename "$data_dir")"
  for _ in $(seq 1 40); do vm true 2>/dev/null && break; sleep 10; done
  stage="$(mktemp -d)"
  (cd "$REPO" && COPYFILE_DISABLE=1 tar --no-xattrs -czf "$stage/code.tgz" \
    fields/__init__.py fields/model.py fields/train.py)
  mkdir -p "$stage/runs"
  while IFS='|' read -r run _; do
    [ -z "$run" ] && continue
    mkdir -p "$stage/runs/$run"
    if [ -f "$RUNS/$run/state.pt" ]; then cp "$RUNS/$run/state.pt" "$stage/runs/$run/"; fi
    if [ -f "$RUNS/$run.log" ]; then cp "$RUNS/$run.log" "$stage/runs/"; fi
  done < "$runs_file"
  COPYFILE_DISABLE=1 tar --no-xattrs -cf "$stage/resume.tar" -C "$stage" runs
  upload "" "$stage/code.tgz" "$stage/resume.tar"
  if vm test -e /dev/disk/by-id/google-lagnav-data; then
    vm "sudo mkdir -p /mnt/data && sudo mount -o ro,noload /dev/disk/by-id/google-lagnav-data /mnt/data \
      && mkdir -p lagnav/data && ln -sfn /mnt/data/$data_name lagnav/data/$data_name \
      && echo \"data disk: \$(ls lagnav/data/$data_name/points | wc -l) subjects\""
  else
    upload data.tar "$(data_tar "$data_dir")"
  fi
  vm 'mkdir -p lagnav/data && cd lagnav && tar -xzf ~/code.tgz 2>/dev/null; tar -xf ~/data.tar -C data 2>/dev/null; tar -xf ~/resume.tar 2>/dev/null; rm -f ~/data.tar'
  while IFS='|' read -r run args; do
    [ -z "$run" ] && continue
    vm "cd ~/lagnav && (PYTHONPATH=~/lagnav setsid nohup python3 -m fields.train --out runs/$run $args >> runs/$run.log 2>&1 < /dev/null &)"
    echo "started $run"
  done < "$runs_file"
  rm -rf "$stage"
  sleep 60
  vm 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; for f in ~/lagnav/runs/*.log; do echo "$(basename "$f"): $(tail -n 1 "$f" | cut -c1-120)"; done'
}

pull() {
  # shellcheck disable=SC1090
  source "$STATE"
  mkdir -p "$RUNS"
  rsync -a -e "ssh -i $KEY -o UserKnownHostsFile=$KNOWN -o BatchMode=yes -o ConnectTimeout=15" \
    "lagnav@${IP}:lagnav/runs/" "$RUNS/"
}

delete_vm() {
  # shellcheck disable=SC1090
  source "$STATE"
  gcloud compute instances delete "$NAME" --project "$PROJECT" --zone "$ZONE" --quiet || true
  echo "VM: $(vm_state || true)"
}

stage() {  # stage DATA_DIR: upload once to a small CPU VM, keep it as a snapshot, clean up
  local data_dir="$1" data_name tar zone=us-central1-a keys disk snapshot ip expected
  data_name="$(basename "$data_dir")"
  disk="lagnav-data-$data_name"
  snapshot="lagnav-$data_name"
  tar="$(data_tar "$data_dir")"
  expected="$(find "$data_dir/points" -name '*.npz' | wc -l | tr -d ' ')"
  STATE="$HOME/.lagnav/stage_state"
  keys="$(mktemp)"
  printf 'lagnav:%s\n' "$(cat "$KEY.pub")" > "$keys"
  gcloud compute instances create lagnav-data-stage --project "$PROJECT" --zone "$zone" \
    --machine-type e2-small --image-family debian-12 --image-project debian-cloud \
    --boot-disk-size 10GB \
    --create-disk "name=$disk,size=10GB,type=pd-standard,device-name=lagnav-data,auto-delete=no" \
    --instance-termination-action DELETE --max-run-duration 6h \
    --no-service-account --no-scopes \
    --metadata-from-file "ssh-keys=$keys" --metadata block-project-ssh-keys=TRUE \
    --labels purpose=lagnav-level2b > /dev/null
  rm -f "$keys"
  ip="$(gcloud compute instances describe lagnav-data-stage --project "$PROJECT" --zone "$zone" \
    --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
  printf 'ZONE=%s\nIP=%s\n' "$zone" "$ip" > "$STATE"
  ssh-keygen -R "$ip" -f "$KNOWN" > /dev/null 2>&1 || true
  echo "stage VM at $ip; uploading $(du -h "$tar" | cut -f1)"
  for _ in $(seq 1 40); do vm true 2>/dev/null && break; sleep 10; done
  vm 'sudo apt-get -qq update > /dev/null && sudo apt-get -qq install -y rsync > /dev/null \
    && sudo mkfs.ext4 -q -F /dev/disk/by-id/google-lagnav-data && sudo mkdir -p /mnt/data \
    && sudo mount /dev/disk/by-id/google-lagnav-data /mnt/data && sudo chown lagnav /mnt/data'
  upload data.tar "$tar"
  local found
  found="$(vm "tar -xf ~/data.tar -C /mnt/data && rm ~/data.tar && ls /mnt/data/$data_name/points | wc -l")"
  vm 'sync && sudo umount /mnt/data'
  if [ "$found" != "$expected" ]; then
    echo "staged $found subjects, expected $expected; keeping VM and disk for inspection" >&2
    return 1
  fi
  gcloud compute instances delete lagnav-data-stage --project "$PROJECT" --zone "$zone" --quiet
  gcloud compute snapshots create "$snapshot" --project "$PROJECT" --source-disk "$disk" \
    --source-disk-zone "$zone" --labels purpose=lagnav-level2b > /dev/null
  gcloud compute disks delete "$disk" --project "$PROJECT" --zone "$zone" --quiet
  echo "snapshot $snapshot holds $found subjects; stage VM and disk deleted"
}

vm_state() {
  gcloud compute instances list --project "$PROJECT" --filter="name=$NAME" \
    --format='value(status)' 2>/dev/null
}

case "${1:-}" in
  launch) launch "$2" "$3" ;;
  deploy) deploy "$2" "$3" ;;
  stage) stage "$2" ;;
  pull) pull ;;
  status)
    echo "VM: $(vm_state || true)"
    vm 'for f in ~/lagnav/runs/*.log; do echo "$(basename "$f"): $(tail -n 1 "$f" | cut -c1-120)"; done' || true
    ;;
  delete) delete_vm ;;
  start)
    # shellcheck disable=SC1090
    source "$STATE"
    gcloud compute instances start "$NAME" --project "$PROJECT" --zone "$ZONE" --quiet
    IP="$(gcloud compute instances describe "$NAME" --project "$PROJECT" --zone "$ZONE" \
      --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
    printf 'ZONE=%s\nIP=%s\n' "$ZONE" "$IP" > "$STATE"
    ssh-keygen -R "$IP" -f "$KNOWN" > /dev/null 2>&1 || true
    echo "started at $IP"
    ;;
  vm) shift; vm "$@" ;;
  *) sed -n '2,26p' "$0"; exit 2 ;;
esac
