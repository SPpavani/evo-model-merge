pip install -r requirements.txt
python src/train_experts.py
python src/core/evolve.py
python src/compare_baselines.py
python -m pytest tests/ -v