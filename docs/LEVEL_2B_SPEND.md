# Level 2b — GPU spend ledger

The budget is USD 700, approved by the user on 2026-10-18. Milestone caps are in
[`LEVEL_2B_PLAN.md`](LEVEL_2B_PLAN.md) §4: M1 $0, M2 ≤ $50, M3 ≤ $100, M4 ≤ $300, reserve
$250 released only by the user. Every session is recorded here the same day, from the
provider's own billing figures. A cap is a hard stop.

| date | provider | GPU | hours | cost (USD) | milestone | purpose | running total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| – | – | – | – | 0.00 | – | nothing spent yet | **0.00** |
| 2026-10-18 | Google Cloud (`lagnever-ai-app-n`), us-central1-a | 1× NVIDIA L4, g2-standard-8 spot | 2.83 (04:44–07:34 UTC) | ~1.00 (estimate; billing figure to replace it) | M2 | VM `lagnav-2b-m2`: benchmark and five training runs; deleted when the user paused | **~1.00** |
| 2026-10-18 | Google Cloud (`lagnever-ai-app-n`), us-east1-b | 1× NVIDIA L4, g2-standard-8 spot | 0.43 (14:05–14:31 UTC) | ~0.15 (estimate) | M2 | second session: runs 4 and 5 resumed; **preempted** by Google at 14:31, VM deleted by its termination action | **~1.15** |
| 2026-10-18 | Google Cloud (`lagnever-ai-app-n`), us-central1-c | 1× NVIDIA L4, g2-standard-8 spot | 1.43 (14:36–16:02 UTC) | ~0.50 (estimate) | M2 | third session: run 4 finished, run 5 to step 28000, run 6 started; **preempted** at 16:02 | **~1.65** |
| 2026-10-18 | Google Cloud (`lagnever-ai-app-n`), us-central1-a | 1× NVIDIA L4, g2-standard-8 spot | 1.47 (16:04–17:32 UTC) | ~0.50 (estimate) | M2 | fourth session: runs 5 and 6 finished; VM deleted | **~2.15** |
| 2026-10-18 | Google Cloud (`lagnever-ai-app-n`), us-central1-a | 1× NVIDIA L4, g2-standard-8 spot | 0.01 (22:56:58–22:57:39 UTC) | ~0.01 | M2/stage 2 | stage 2 chest runs A and B, approved by the user (≤ $5): **preempted 41 s after creation**, before training began; the user then paused for the night | **~2.16** |
| 2026-10-19 | Google Cloud (`lagnever-ai-app-n`), us-central1-a | 1× NVIDIA L4, g2-standard-8 spot | 0.23 + 0.28 (09:19–09:33, 09:35–09:52 UTC) | ~0.20 (estimate) | stage 2 | chest runs A and B; **preempted twice**, saved at step 2000 | **~2.36** |
| 2026-10-19 | Google Cloud (`lagnever-ai-app-n`), us-central1-b | 1× NVIDIA L4, g2-standard-8 **on-demand** | 3.34 (09:55–13:16 UTC) | ~2.85 (estimate, ~$0.85/h) | stage 2 | after two preemptions, per the user's rule: runs A and B finished; VM deleted | **~5.20** |

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

### 2026-10-18 · M2 second GPU session · `lagnav-2b-m2`

- **Created** at 14:05 UTC, after the user's explicit yes to the plan and its estimate:
  about 3 h and $0.75–1.35.
- **Zone:** spot L4 capacity was exhausted in us-central1-a, -b and -c, so the VM was created
  in us-east1-b, where the project's L4 quota also applies.
- **Same machine and safeguards as the first session:** 6 h auto-delete, no service account
  or scopes, instance-only SSH key, and checkpoints pulled to the Mac every 4 minutes.
- **Preempted at 14:31 UTC** (`compute.instances.preempted`), after 26 minutes. The
  termination action deleted the VM, so nothing was left billing. The Mac's last pull, at
  14:27, held run 4 at step 12,000 and run 5 at step 6,000; run 5 lost about 1000 steps.
- **The user was asked again,** with spot (~$0.6–1.1 for the remaining ~2.5 h) and on-demand
  (~$2–2.5, never reclaimed) as the options. They chose spot.
- **Third VM** created at 14:36 UTC in us-central1-c. Five zones had no spot L4 capacity
  first: us-east1-b/c/d and us-central1-a/b. Both runs resumed from their saved steps. Launch,
  upload and resume are now one script, so a further preemption costs minutes.
- **Preempted again at 16:02 UTC,** after 1.43 h. By then run 4 had finished and its
  checkpoint was on the Mac. Run 5 was saved at step 28,000 and run 6 at step 2,000.
- **Fourth VM** created at 16:04 UTC in us-central1-a, after us-east1-b/c/d had no capacity.
  It is covered by the user's spot choice, whose stated terms were "may be preempted again;
  resumes each time". Spend stays far under the M2 cap: about $1.65 so far, against $50.
- **Fourth session finished.** Runs 5 and 6 completed. Every run was pulled to the Mac and
  run 6's checkpoint checked before the VM was deleted at 17:32 UTC; no lagnav instance or
  disk remains.
  - **Bug in the launcher:** `ssh` inside a `while read` loop consumed the run list, so run 6
    did not start until 16:18. Fixed with `ssh -n`.
  - **Day total:** about 6.2 VM-hours across four spot sessions, two of them preempted, for
    an estimated **~$2.15**. M2's cap is $50.

### 2026-10-18 · Stage 2 attempt · preempted at once, then paused

- The user approved a spot session for the stage 2 chest runs, capped at $5.
- The VM was created at 22:56:58 UTC in us-central1-a, after us-east1-b/c/d had no
  capacity. Google preempted it 41 s later, before the upload finished.
- The user then asked to pause for the night. The launcher was stopped, and a check confirmed
  that no lagnav instance or disk remains. Cost: about $0.01.

### 2026-10-19 · Stage 2 chest runs

- **Approval:** the user chose "spot, fallback on-demand". Spot is tried first. After two
  preemptions, or if no zone has spot capacity, the session moves to on-demand. The cap
  for the session is $5.
- **Launch:** created at 09:19 UTC in us-central1-a by the new `tools/gpu_session.sh`,
  after us-east1-b/c/d had no spot capacity. Runs A and B started together and use 12 GB
  of the L4.
- **Preempted twice** (09:33 and 09:52 UTC). The first replacement reused the same IP; the
  launcher now handles that. As the user's rule says, the session moved to on-demand: a VM
  created at 09:55 UTC in us-central1-b, after four zones had no on-demand capacity either.
  Both runs resumed from step 2000. Spend stays under the $5 session cap.
- **Finished.** Both runs completed 40,000 steps at 13:14 UTC. They were pulled and their
  checkpoints checked before the VM was deleted at 13:15:56; no lagnav instance or disk
  remains.
- **Stage 2 session:** about $3.05, under its $5 cap. **Project total: about $5.20** of
  the $700 budget.
