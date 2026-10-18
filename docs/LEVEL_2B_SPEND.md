# Level 2b — GPU spend ledger

The budget is USD 700, approved by the user on 2026-10-18. Milestone caps are in
[`LEVEL_2B_PLAN.md`](LEVEL_2B_PLAN.md) §4: M1 $0, M2 ≤ $50, M3 ≤ $100, M4 ≤ $300, reserve
$250 released only by the user. Every session is recorded here the same day, from the
provider's own billing figures. A cap is a hard stop.

| date | provider | GPU | hours | cost (USD) | milestone | purpose | running total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| – | – | – | – | 0.00 | – | nothing spent yet | **0.00** |
| 2026-10-18 | Google Cloud (`lagnever-ai-app-n`), us-central1-a | 1× NVIDIA L4, g2-standard-8 spot | 2.83 (04:44–07:34 UTC) | ~1.00 (estimate; billing figure to replace it) | M2 | VM `lagnav-2b-m2`: benchmark and five training runs; deleted when the user paused | **~1.00** |

## Sessions

### 2026-10-18 · M2 first GPU run · `lagnav-2b-m2`

- **Machine:** g2-standard-8 (8 vCPU, 32 GB) with 1× NVIDIA L4 (24 GB). Spot provisioning;
  zone us-central1-a. Image `pytorch-2-9-cu129-ubuntu-2404-nvidia-580-v20261001` (PyTorch
  2.9.1, CUDA 12.9, driver 580). 100 GB balanced boot disk.
- **Safeguards:**
  - `--max-run-duration 6h` with termination action DELETE: the VM deletes itself at six
    hours whatever happens.
  - No service account and no API scopes, so the VM holds no credentials to the project
    (which also hosts production Firebase and user uploads).
  - SSH by a dedicated key in instance metadata, with project keys blocked; project
    metadata unchanged.
  - The Mac pulls checkpoints, so a preemption loses at most one save interval.
- **Created:** 2026-10-18 04:44 UTC by `gcloud compute instances create`, after the user's
  explicit yes to the plan and its estimate (~$0.25–0.45 per hour spot; ~$1–3 in total).
- **Deleted:** 07:34 UTC with `gcloud compute instances delete`, when the user paused for
  the day. All runs were first pulled to the Mac and checked. Afterwards, no instance and
  no disk named lagnav remain in the project.
- **Estimate:** 2.83 h × ~$0.25–0.45/h spot, plus disk and IP, comes to about $0.75–1.35.
  It will be replaced by Google's billing figure once it posts. M2 cap: $50.
