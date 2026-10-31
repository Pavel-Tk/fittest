cd /content/trustswarm && python3 -m pytest tests/test_hook.py -q -rs -p no:cacheprovider 2>&1 | tail -40
