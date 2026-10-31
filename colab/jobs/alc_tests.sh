cd /content/trustswarm && python3 -m pytest tests/test_hook.py -q -rs -p no:cacheprovider 2>&1 | tail -40; python3 scripts/_perf_mace.py 2>&1 | grep -v Warning | tail -5
