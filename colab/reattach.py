from colab_cli.state import SessionState
from colab_cli.common import *  # noqa
import colab_cli.common as c
st = [v for v in vars(c).values() if isinstance(v, type) and hasattr(v, "get_session")]
print("state classes:", st)
from colab_cli import state as s_mod
store_cls = [v for v in vars(s_mod).values() if isinstance(v, type) and hasattr(v, "add") and hasattr(v, "list")]
print("store classes:", store_cls)
store = store_cls[0]() if store_cls else None
s = SessionState(name="tswarm", token="", url="", endpoint="gpu-l4-s-ass1b0-2wu2ne3a66g3h",
                 variant="GPU", accelerator="L4", machine_shape="STANDARD")
store.add(s); print("added", store.list().keys())
