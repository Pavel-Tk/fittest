/home/pavel/.local/bin/uv tool install --force "google-colab-cli==0.7.5" --with "jupyter-kernel-client<1.0" 2>&1 | tail -2
/home/pavel/.local/bin/colab version 2>&1 | tail -1
/home/pavel/.local/bin/colab --auth adc help 2>&1 | grep -i -E "attach|adopt|resume|reconnect" 
/home/pavel/.local/bin/colab --auth adc sessions 2>&1 | tail -2
ls ~/.config/colab-cli/; python3 -c "import json;d=json.load(open('/home/pavel/.config/colab-cli/sessions.json'));print(json.dumps(d)[:600])" 2>&1
