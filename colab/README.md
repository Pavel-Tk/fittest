# Driving the Colab GPU (session `tswarm`, NVIDIA L4 24 GB, CUDA 13.0 driver)

Everything is run from the laptop with Git Bash; the VM is a shared Jupyter kernel.

| Need | Command (from repo root, Git Bash) |
|---|---|
| Push local code to `/content/fittest` | `colab/sync.sh` (tars fittest/ scripts/ tests/ pyproject.toml; concurrency-safe) |
| Run a short (<60 s) python file in the kernel | `colab/cx.sh path/to/file.py 300` |
| Start a detached job (anything longer) | `powershell -Command "wsl -d Ubuntu --cd /home/pavel -- bash /mnt/c/Software_projects/Workshops/science_claw/fittest/colab/jc.sh send -s tswarm start <name> --body-file /mnt/c/<path to .sh>"` |
| Job status + log tail | same with `status <name> --tail 40` |
| Block until done | DO NOT use `wait`: it occupies the shared kernel and blocks every other agent. Poll `status` every few minutes instead (or loop locally: sleep between status calls in a background bash). |
| Get a file back | write a tiny cx.sh python that prints it, or base64 small files; for big results tar on VM then `MSYS_NO_PATHCONV=1 powershell -File "C:/Users/ptkac/.claude/plugins/cache/colab-cli-ops/colab-cli-ops/0.1.0/skills/colab-cli-ops/scripts/colab.ps1" download -s tswarm /content/x.tgz C:\\...\\x.tgz` |

Rules
* The kernel is **shared by several agents and serialises execs**: keep `cx.sh` calls short. Run real work as detached jobs (`jc.sh start`) that call `python3 scripts/...` in a subprocess, never inside the kernel.
* Never import project modules into the kernel for long work; if you do for a quick check, purge `sys.modules` entries starting with `fittest` first (see colab/smoke2.py).
* Python environments on the VM:
  * core: system `/usr/bin/python3` (3.13): torch 2.11+cu130, nvalchemi-toolkit 0.3.0 (rc branch, editable at /content/nvalchemi-toolkit), mace-torch 0.3.15, aimnet 0.2.5, ase, h5py, sklearn, fastapi.
  * orb: `/content/envs/orb/bin/python` (3.12): orb-models (old 0.5-style API), torch 2.15. Only for the Orb model server.
* Data: `/content/data/Transition1x.h5` (6.6 GB, md5 verified). Write outputs under `/content/fittest/data/` and `/content/fittest/results/`, logs under `/content/logs/`.
* Model servers (`fittest/model_server.py`): `spawn(model, port, python=...)` + `.connect()`. Port ranges so agents don't collide: lead 6000–6019, ALCHEMI agent 6100–6119, fine-tune agent 6200–6219.
* GPU is shared (24 GB). Keep batch sizes modest; free GPU memory when done.
* Do not stop the session and do not `colab new`.
