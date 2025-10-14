import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.models.ikd_model import IKDModel


def make_predictions(model_path, data_path, plot_title):
    """
    Make predictions using the trained IKD model on test data.
    
    Args:
        model_path: Path to the saved model weights
        data_path: Path to the CSV data file
        plot_title: Title for the plot
    """
    # Load model
    model = IKDModel(2, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load data
    data = pd.read_csv(data_path)
    joystick = np.array([eval(i) for i in data["joystick"]])
    executed = np.array([eval(i) for i in data["executed"]])
    data = np.concatenate((joystick, executed), axis=1)
    
    actual_av = []
    predicted_av = []
    time = []
    
    # Make predictions
    with torch.no_grad():
        for i, row in enumerate(data):
            time.append(i)
            v = row[0]
            av = row[1]
            true_av = row[2]
            
            # Model takes velocity and true angular velocity as input
            input_tensor = torch.FloatTensor([v, true_av])
            output = model(input_tensor)
            
            predicted_av.append(output.item())
            actual_av.append(av)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, predicted_av, label='Predicted Angular Velocity', linewidth=2)
    ax.plot(time, actual_av, label='Commanded Angular Velocity', linewidth=2, alpha=0.7)
    ax.legend()
    ax.set_title(plot_title)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Available test datasets
    test_datasets = {
        'loose_ccw': './dataset/loose_ccw.csv',
        'loose_cw': './dataset/loose_cw.csv',
        'tight_cw': './dataset/tight_cw.csv',
        'tight_ccw': './dataset/tight_ccw.csv'
    }
    
    model_path = './ikddata2.pt'
    
    # Run predictions on all test datasets
    for name, path in test_datasets.items():
        print(f"\n[INFO] Running predictions on {name}...")
        try:
            make_predictions(model_path, path, f"Predictions: {name.replace('_', ' ').title()}")
        except FileNotFoundError:
            print(f"[WARNING] Dataset not found: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to process {name}: {str(e)}")
