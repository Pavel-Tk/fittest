# Our Colab L4 layout (source this on the VM; other machines: set the same variables yourself).
export TRUSTSWARM_HOME=${TRUSTSWARM_HOME:-/content/trustswarm}
export TRUSTSWARM_T1X=${TRUSTSWARM_T1X:-/content/data/Transition1x.h5}
export TRUSTSWARM_CORE_PYTHON=${TRUSTSWARM_CORE_PYTHON:-/usr/bin/python3}
export TRUSTSWARM_ORB_PYTHON=${TRUSTSWARM_ORB_PYTHON:-/content/envs/orb/bin/python}
export TRUSTSWARM_NEB_DIR=${TRUSTSWARM_NEB_DIR:-/content/nvalchemi-toolkit/examples/advanced/batched_neb_endpoints}
export TRUSTSWARM_LOG_DIR=${TRUSTSWARM_LOG_DIR:-/content/logs}
cd "$TRUSTSWARM_HOME" && mkdir -p results data "$TRUSTSWARM_LOG_DIR"
