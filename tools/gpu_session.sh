#!/bin/bash
# Level 2b GPU sessions on Google Cloud: one L4 VM with the safeguards in docs/LEVEL_2B_SPEND.md.
#
#   tools/gpu_session.sh launch RUNS_FILE DATA_DIR   create the VM, upload, resume or start runs
#   tools/gpu_session.sh deploy RUNS_FILE DATA_DIR   the same onto the VM that already exists
#   tools/gpu_session.sh pull                       copy every run's files back to the Mac
#   tools/gpu_session.sh status                     VM state and the last line of each run log
#   tools/gpu_session.sh delete                     delete the VM
#
# RUNS_FILE has one run per line, "name|fields.train arguments" (tools/gpu_runs/*.txt).
# DATA_DIR is a fields.data output folder (manifest.json and points/).
# PROVISIONING=SPOT (default) or STANDARD (on-demand, never reclaimed).
#
# Safeguards: the VM deletes itself after 6 hours; it has no service account and no API
# scopes; SSH uses a dedicated key in instance metadata with project keys blocked. A run
# resumes from the newest state.pt on the Mac, so losing the VM costs one save interval.
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

create() {
  local provisioning="${PROVISIONING:-SPOT}" keys
  keys="$(mktemp)"
  printf 'lagnav:%s\n' "$(cat "$KEY.pub")" > "$keys"
  for zone in $ZONES; do
    if gcloud compute instances create "$NAME" --project "$PROJECT" --zone "$zone" \
        --machine-type g2-standard-8 --provisioning-model "$provisioning" \
        --instance-termination-action DELETE --max-run-duration 6h \
        --maintenance-policy TERMINATE \
        --image-family pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
        --image-project deeplearning-platform-release \
        --boot-disk-size 100GB --boot-disk-type pd-balanced \
        --no-service-account --no-scopes \
        --metadata-from-file "ssh-keys=$keys" \
        --metadata block-project-ssh-keys=TRUE,install-nvidia-driver=True \
        --labels purpose=lagnav-level2b,milestone=m2 > /dev/null 2>&1; then
      local ip
      ip="$(gcloud compute instances describe "$NAME" --project "$PROJECT" --zone "$zone" \
        --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
      printf 'ZONE=%s\nIP=%s\n' "$zone" "$ip" > "$STATE"
      # A new VM can reuse a released IP with a new host key; forget the old one.
      ssh-keygen -R "$ip" -f "$KNOWN" > /dev/null 2>&1 || true
      rm -f "$keys"
      echo "created $provisioning in $zone at $(date -u +%H:%M:%S) UTC"
      return 0
    fi
    echo "no $provisioning capacity in $zone"
  done
  rm -f "$keys"
  echo "no zone had $provisioning capacity" >&2
  return 1
}

launch() {
  create
  deploy "$1" "$2"
}

deploy() {
  local runs_file="$1" data_dir="$2" data_name stage
  data_name="$(basename "$data_dir")"
  mkdir -p "$CACHE"
  if [ ! -f "$CACHE/$data_name.tar" ] || [ "$data_dir/manifest.json" -nt "$CACHE/$data_name.tar" ]; then
    COPYFILE_DISABLE=1 tar --no-xattrs -cf "$CACHE/$data_name.tar" \
      -C "$(dirname "$data_dir")" "$data_name/manifest.json" "$data_name/points"
  fi
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
  # shellcheck disable=SC1090
  source "$STATE"
  scp -q -i "$KEY" -o UserKnownHostsFile="$KNOWN" -o BatchMode=yes \
    "$stage/code.tgz" "$stage/resume.tar" "lagnav@${IP}:~/"
  scp -q -i "$KEY" -o UserKnownHostsFile="$KNOWN" -o BatchMode=yes \
    "$CACHE/$data_name.tar" "lagnav@${IP}:~/data.tar"
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

vm_state() {
  gcloud compute instances list --project "$PROJECT" --filter="name=$NAME" \
    --format='value(status)' 2>/dev/null
}

case "${1:-}" in
  launch) launch "$2" "$3" ;;
  deploy) deploy "$2" "$3" ;;
  pull) pull ;;
  status)
    echo "VM: $(vm_state || true)"
    vm 'for f in ~/lagnav/runs/*.log; do echo "$(basename "$f"): $(tail -n 1 "$f" | cut -c1-120)"; done' || true
    ;;
  delete)
    # shellcheck disable=SC1090
    source "$STATE"
    gcloud compute instances delete "$NAME" --project "$PROJECT" --zone "$ZONE" --quiet || true
    echo "VM: $(vm_state || true)"
    ;;
  vm) shift; vm "$@" ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
