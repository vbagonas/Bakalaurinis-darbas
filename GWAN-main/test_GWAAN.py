import sys
import torch
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, './GWAN-main/GWAN_master')
from models2.GWAN import GWAN
from torch_geometric.loader import DataLoader

# ── Load test graphs ──────────────────────────────────────────────────────────
with open('./GWAN-main/GWAN_master/datasets/HypoidGearDWTV2_test_batch2.pkl', 'rb') as f:
    test_graphs = pickle.load(f)
print(f'Test graphs: {len(test_graphs)}')

test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

# ── Load model ────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GWAN(feature=1024, out_channel=7).to(DEVICE)
model.load_state_dict(torch.load("C:/Users/bagon/OneDrive/Desktop/TactisRealization/checkpoint/Graph_GWAN_HypoidGearRadius_TD_0420-153936/95-0.7189-best_model.pth", map_location=DEVICE))
model.eval()

# ── Evaluate ──────────────────────────────────────────────────────────────────
all_preds, all_labels = [], []
with torch.no_grad():
    for data in test_loader:
        data = data.to(DEVICE)
        preds = model(data).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(data.y.cpu().numpy())

print(f'Test accuracy: {accuracy_score(all_labels, all_preds):.4f}')
print(classification_report(all_labels, all_preds, digits=4))