import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
import os
import copy

# Define the dataset path
CSV_FILE = 'calculator_data.csv'

# Operation mapping to one-hot indices: p, s, m, d
# We have 4 operations.
OP_TO_IDX = {'p': 0, 's': 1, 'm': 2, 'd': 3}

def load_data(file_path):
    df = pd.read_csv(file_path)
    X = []
    Y = []
    for index, row in df.iterrows():
        var1 = float(row['Var1'])
        op_str = str(row['op']).strip()
        var2 = float(row['Var2'])
        result = float(row['Result'])
        
        # one hot encoding for operations
        op_vec = [0.0, 0.0, 0.0, 0.0]
        if op_str in OP_TO_IDX:
            op_vec[OP_TO_IDX[op_str]] = 1.0
            
        feature = [var1] + op_vec + [var2]
        X.append(feature)
        Y.append([result])
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

class CalcNet(nn.Module):
    def __init__(self):
        super(CalcNet, self).__init__()
        # 1 (Var1) + 4 (ops) + 1 (Var2) = 6 input features
        # Using 32 neurons per layer for much better accuracy
        self.fc1 = nn.Linear(6, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 32)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.out(x)

def extract_weights(model):
    """Extract all weights and biases as plain Python lists."""
    return {
        'fc1_w': model.fc1.weight.detach().numpy().tolist(),
        'fc1_b': model.fc1.bias.detach().numpy().tolist(),
        'fc2_w': model.fc2.weight.detach().numpy().tolist(),
        'fc2_b': model.fc2.bias.detach().numpy().tolist(),
        'fc3_w': model.fc3.weight.detach().numpy().tolist(),
        'fc3_b': model.fc3.bias.detach().numpy().tolist(),
        'out_w': model.out.weight.detach().numpy().tolist(),
        'out_b': model.out.bias.detach().numpy().tolist()
    }

def train():
    print("Loading data...")
    X, Y = load_data(CSV_FILE)
    
    print(f"Loaded {len(X)} samples.")
    model = CalcNet()
    criterion = nn.MSELoss()

    # -------------------------------------------------------
    # PHASE 1: Capture training timeline snapshots for tutorial
    # -------------------------------------------------------
    print("\n=== PHASE 1: Capturing training evolution snapshots ===")
    
    # Snapshot 0: Random initial weights (before any training)
    snapshot_initial = extract_weights(model)
    
    # Snapshot 1: Train on JUST the first row
    optimizer_single = optim.Adam(model.parameters(), lr=0.01)
    row1_x = X[0:1]  # first row only
    row1_y = Y[0:1]
    
    # Train 50 steps on just row 1 so you can see weights shift
    for _ in range(50):
        optimizer_single.zero_grad()
        out = model(row1_x)
        loss = criterion(out, row1_y)
        loss.backward()
        optimizer_single.step()
    
    snapshot_row1 = extract_weights(model)
    row1_loss = criterion(model(row1_x), row1_y).item()
    print(f"  After Row 1 training (50 steps): Loss = {row1_loss:.4f}")
    
    # Snapshot 2: Now also train on row 2
    row2_x = X[1:2]
    row2_y = Y[1:2]
    for _ in range(50):
        optimizer_single.zero_grad()
        out = model(row2_x)
        loss = criterion(out, row2_y)
        loss.backward()
        optimizer_single.step()
    
    snapshot_row2 = extract_weights(model)
    row2_loss = criterion(model(row2_x), row2_y).item()
    print(f"  After Row 2 training (50 steps): Loss = {row2_loss:.4f}")
    
    # Build the training samples info for the tutorial
    row1_info = {"input": X[0].tolist(), "expected": Y[0].item()}
    row2_info = {"input": X[1].tolist(), "expected": Y[1].item()}

    # -------------------------------------------------------
    # PHASE 2: Full training from scratch for the main model
    # -------------------------------------------------------
    print("\n=== PHASE 2: Full training (fresh model, 20,000 epochs) ===")
    
    # Normalize outputs so the network doesn't learn from raw result scale.
    # Normalization helps the model train faster and more reliably.
    y_mean = Y.mean()
    y_std = Y.std()
    Y_norm = (Y - y_mean) / y_std
    
    model_full = CalcNet()
    optimizer = optim.Adam(model_full.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    
    epochs = 20000
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model_full(X)
        loss = criterion(outputs, Y_norm)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    snapshot_final = extract_weights(model_full)
    
    # Quick accuracy test
    with torch.no_grad():
        test_pred = model_full(X)
        test_pred_real = test_pred * y_std + y_mean
        test_mse = criterion(test_pred_real, Y).item()
        print(f"\n  Final Test MSE (real scale): {test_mse:.4f}")
        print(f"  Final Test RMSE: {test_mse**0.5:.4f}")
    
    # -------------------------------------------------------
    # Save main model weights (used by calculator page)
    # -------------------------------------------------------
    print("\nExporting model_weights.json...")
    metadata = {
        "description": "Simple Neural Network Calculator Model",
        "instructions": "To use this model, pass a 1x6 input vector through the weights using Dot Products + ReLU. The raw output is normalized — multiply by y_std and add y_mean to get the real prediction.",
        "architecture": "Input[6] -> Linear[32] -> ReLU -> Linear[32] -> ReLU -> Linear[32] -> ReLU -> Linear[1] -> Output",
        "operators_encoding": {"+": [1,0,0,0], "-": [0,1,0,0], "*": [0,0,1,0], "/": [0,0,0,1]},
        "y_mean": y_mean.item(),
        "y_std": y_std.item()
    }
    
    weights_dict = {
        '__metadata__': metadata,
        **snapshot_final
    }
    
    with open('model_weights.json', 'w') as f:
        json.dump(weights_dict, f, indent=4)
    
    # -------------------------------------------------------
    # Save training timeline (used by tutorial page)
    # -------------------------------------------------------
    print("Exporting training_timeline.json...")
    timeline = {
        "training_rows": {
            "row1": row1_info,
            "row2": row2_info
        },
        "snapshots": {
            "initial": {
                "label": "Random Initialization (Before Training)",
                "description": "These are the completely random numbers PyTorch assigned when the model was first created. They mean nothing yet!",
                **snapshot_initial
            },
            "after_row1": {
                "label": "After Training on Row 1",
                "description": f"We fed the first training example ({row1_info['input']}) with expected answer {row1_info['expected']} through the network 50 times. Watch how the weights shifted!",
                **snapshot_row1
            },
            "after_row2": {
                "label": "After Training on Row 2", 
                "description": f"We then fed the second training example ({row2_info['input']}) with expected answer {row2_info['expected']}. The weights shifted again to accommodate both examples!",
                **snapshot_row2
            },
            "final": {
                "label": "After Full Training (10,000 rows × 20,000 epochs)",
                "description": "After seeing the entire dataset thousands of times, the weights have sculpted themselves into a precise calculator. Compare these to the random initial values above!",
                **snapshot_final
            }
        }
    }
    
    with open('training_timeline.json', 'w') as f:
        json.dump(timeline, f, indent=4)
        
    print("Export complete! Files: model_weights.json, training_timeline.json")

if __name__ == '__main__':
    train()
