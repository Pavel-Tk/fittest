P=$(ls -d /home/pavel/.local/share/uv/tools/google-colab-cli/lib/python3.*/site-packages/colab_cli)
sed -n 60,160p $P/common.py
grep -n "def refresh_proxy\|def get_runtime_proxy\|runtime-proxy-token\|def keep_alive\|def tun" $P/client.py | head
