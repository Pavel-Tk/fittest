source /content/fittest/colab/env.sh
export PYTHONPATH=/content/fittest
python3 scripts/demo_service.py --snapshot results/trust_snapshot.json --n 12
# HTTP smoke test
python3 -m fittest.service serve --snapshot results/trust_snapshot.json --port 8000 --base-port 6450 --orb-python $FITTEST_ORB_PYTHON > /content/logs/service_http.log 2>&1 &
SP=$!
for i in $(seq 1 90); do curl -s localhost:8000/health && break; sleep 2; done; echo
curl -s -X POST localhost:8000/trust -H 'content-type: application/json' -d '{"systems":[{"name":"water","Z":[8,1,1],"R":[[0,0,0],[0.96,0,0],[-0.24,0.93,0]]}]}' | python3 -c "import json,sys; d=json.load(sys.stdin)['results'][0]; d.pop('served_forces'); print(json.dumps(d))"
kill $SP; sleep 3; pkill -f "fittest.model_server --model" || true
echo SERVICE_DONE
