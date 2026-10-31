cd /content/trustswarm && mkdir -p results /content/logs && python3 scripts/check_mace_server.py --port 6100 --n-reactions 5 --n-path 2 --log-dir /content/logs/alc 2>&1
