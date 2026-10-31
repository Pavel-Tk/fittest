import os, shutil, subprocess
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout.strip() or "NO GPU")
print(subprocess.run(["nvidia-smi"],capture_output=True,text=True).stdout.split("\n")[2])
print(f"disk free /content: {shutil.disk_usage('/content').free/1e9:.0f} GB   cpus: {os.cpu_count()}")
import sys; print(sys.version)
print(subprocess.run("free -g; which uv; python -c 'import torch;print(torch.__version__, torch.version.cuda)'",shell=True,capture_output=True,text=True).stdout)
