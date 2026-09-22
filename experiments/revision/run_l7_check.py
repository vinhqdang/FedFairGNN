import os
import sys
import time
import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath("."))
from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer

t0 = time.perf_counter()
cfg = ExperimentConfig.canonical(dataset="german", seed=42)
trainer = FederatedTrainer(cfg)
res = trainer.run()
t1 = time.perf_counter()

auc = res["final"]["auc"]
dpd = res["final"]["dpd_hard"]
print(f"wall={t1-t0:.2f}s | AUC={auc:.10f} | DPD_hard={dpd:.10f} | threads={torch.get_num_threads()}")
