import torch
import numpy as np
from model import BalancedDeepModel



def load_model(model_path='best_model.pth'):
    model = BalancedDeepModel(output_dim=8)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def predict(model, X_input, device):
    model.to(device)
    if isinstance(X_input, list):
        X_input = np.array(X_input)
    X_input = torch.from_numpy(X_input).float().to(device)
    with torch.no_grad():
        outputs = model(X_input)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

def main_predict(X_input):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model()
    predictions = predict(model, X_input, device)
    
    return predictions
