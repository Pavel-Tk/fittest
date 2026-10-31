import subprocess
print(subprocess.run(r"sed -n '1,7p;40,140p' /content/logs/install.log | cut -c1-200", shell=True, capture_output=True, text=True).stdout)
print(subprocess.run("curl -sIL -A 'Mozilla/5.0' https://figshare.com/ndownloader/files/36035789 | grep -i -E '^HTTP|location|content-length' | head", shell=True, capture_output=True, text=True).stdout)
