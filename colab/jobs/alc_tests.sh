. /content/trustswarm/colab/env.sh && $TRUSTSWARM_CORE_PYTHON -m pytest tests/test_hook.py -q -rs -p no:cacheprovider 2>&1 | tail -40
