cd /content/fittest
python3 scripts/predict_pool.py --models mace_mpa0 --port 6010 --batch-atoms 2048
python3 scripts/analyze_signal.py --frames data/frames.pkl --preds data/preds --out results
