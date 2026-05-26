import torch
import argparse
import sys
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch_geometric.loader import DataLoader
from models2.GWAN import GWAN

with open('./GWAN-main/GWAN_master/datasets/HypoidGearRadius_test.pkl', 'rb') as f:
    test_graphs = pickle.load(f)
print(f'Test graphs: {len(test_graphs)}')

test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

checkpoint = torch.load('C:/Users/bagon/OneDrive/Desktop/TactisRealization/checkpoint/Graph_GWAN_HypoidGearRadius_TD_0413-121939/95-0.7630-best_model.pth')

model = GWAN(args).cuda()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        data = data.cuda()
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

print(f'Test accuracy: {correct/total*100:.2f}%')