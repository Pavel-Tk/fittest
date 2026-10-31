import subprocess
def sh(c, t=600):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); print(r.stdout[-3000:], r.stderr[-2000:])
sh("python3 -c 'import nvalchemi, torch, mace, aimnet; print(\"CORE OK\", torch.__version__, torch.cuda.is_available())'")
for u in ["https://api.figshare.com/v2/file/download/36035789", "https://figshare.com/ndownloader/files/36035789?private_link=", "https://ndownloader.figshare.com/files/36035789"]:
    sh(f"curl -sIL -A 'Mozilla/5.0 (X11; Linux x86_64)' '{u}' | grep -i -E '^HTTP|^location|content-length' | head -6")
sh("curl -s https://api.figshare.com/v2/articles/19614657 | head -c 1500")
