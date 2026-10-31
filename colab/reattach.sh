PY=$(ls /home/pavel/.local/share/uv/tools/google-colab-cli/bin/python)
$PY /mnt/c/Software_projects/Workshops/science_claw/trustswarm/colab/reattach.py
/home/pavel/.local/bin/colab --auth adc sessions 2>&1 | tail -2
