# Our Colab L4 layout (source this on the VM; other machines: set the same variables yourself).
export FITTEST_HOME=${FITTEST_HOME:-/content/fittest}
export FITTEST_T1X=${FITTEST_T1X:-/content/data/Transition1x.h5}
export FITTEST_CORE_PYTHON=${FITTEST_CORE_PYTHON:-/usr/bin/python3}
export FITTEST_ORB_PYTHON=${FITTEST_ORB_PYTHON:-/content/envs/orb/bin/python}
export FITTEST_NEB_DIR=${FITTEST_NEB_DIR:-/content/nvalchemi-toolkit/examples/advanced/batched_neb_endpoints}
export FITTEST_LOG_DIR=${FITTEST_LOG_DIR:-/content/logs}
cd "$FITTEST_HOME" && mkdir -p results data "$FITTEST_LOG_DIR"
