set -e
cd /content/trustswarm
export PYTHONPATH=/content/trustswarm
echo "== stage 1: ablation (records Curator request)"; date
python3 scripts/run_ablation.py --frames data/frames.pkl --preds data/preds --out results/stage1 --n-bootstrap 200
cat results/stage1/oracle_labels_armD.json | head -c 300; echo
echo "== stage 2: Curator LoRA repair on the requested labels"; date
python3 scripts/finetune.py --frames data/frames.pkl --labelled-uids results/stage1/oracle_labels_armD.json \
  --model mace_mpa0 --out data/preds/mace_lora_curated.npz --ckpt data/ckpt/mace_lora_curated --epochs 40 --seed 0 --predict-uids all
echo "== stage 3: final ablation with repair replayed + budget sweep"; date
python3 scripts/run_ablation.py --frames data/frames.pkl --preds data/preds --out results --budgets 20,40,80
echo "== stage 4: LoRA seed ensemble (B')"; date
python3 scripts/finetune.py --frames data/frames.pkl --labelled-uids results/stage1/oracle_labels_armD.json \
  --model mace_mpa0 --out data/preds/mace_lora_curated.npz --ckpt data/ckpt/mace_seed --epochs 40 --seeds 0,1,2,3 --predict-uids all
python3 scripts/run_ablation.py --frames data/frames.pkl --preds data/preds --out results/with_bprime --n-bootstrap 1000
echo REAL_DONE; date
