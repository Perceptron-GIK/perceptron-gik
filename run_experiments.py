#!/usr/bin/env python3
"""
GIK Model Experiments - Test different configurations to improve training accuracy.

This script tests various hyperparameter combinations and reports the best one.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
import numpy as np
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pretraining import preprocess_and_export, load_preprocessed_dataset
from ml.models.basic_nn import create_model_from_dataset, GIKTrainer

torch.manual_seed(42)
np.random.seed(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Data paths
DATA_DIR = "data_hazel_1"
KEYBOARD_FILE = "Keyboard_1.csv"
LEFT_FILE = "Left_1.csv"
RIGHT_FILE = None

# Configurations to test
EXPERIMENTS = [
    {
        'name': 'baseline',
        'config': {
            'max_seq_length': 10,
            'hidden_dim': 8,
            'num_layers': 1,
            'bidirectional': False,
            'dropout': 0.6,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'epochs': 150,
            'early_stopping': 20,
            'model_type': 'gru',
        }
    },
    {
        'name': 'config_A_lstm_32',
        'config': {
            'max_seq_length': 10,
            'hidden_dim': 32,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.3,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'epochs': 150,
            'early_stopping': 20,
            'model_type': 'lstm',
        }
    },
    {
        'name': 'config_B_lstm_64',
        'config': {
            'max_seq_length': 10,
            'hidden_dim': 64,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.2,
            'batch_size': 32,
            'learning_rate': 5e-4,
            'epochs': 150,
            'early_stopping': 20,
            'model_type': 'lstm',
        }
    },
    {
        'name': 'config_C_gru_64',
        'config': {
            'max_seq_length': 10,
            'hidden_dim': 64,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.25,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'epochs': 150,
            'early_stopping': 20,
            'model_type': 'gru',
        }
    },
    {
        'name': 'config_D_lstm_128',
        'config': {
            'max_seq_length': 10,
            'hidden_dim': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.3,
            'batch_size': 32,
            'learning_rate': 5e-4,
            'epochs': 150,
            'early_stopping': 20,
            'model_type': 'lstm',
        }
    },
    {
        'name': 'config_E_transformer',
        'config': {
            'max_seq_length': 10,
            'hidden_dim': 32,
            'num_layers': 2,
            'bidirectional': False,  # not used for transformer
            'dropout': 0.2,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'epochs': 150,
            'early_stopping': 20,
            'model_type': 'transformer',
        }
    },
    {
        'name': 'config_F_cnn',
        'config': {
            'max_seq_length': 10,
            'hidden_dim': 64,
            'num_layers': 1,  # not used for CNN
            'bidirectional': False,  # not used for CNN
            'dropout': 0.3,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'epochs': 150,
            'early_stopping': 20,
            'model_type': 'cnn',
        }
    },
]


def run_experiment(exp_name: str, config: dict) -> dict:
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*60}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Preprocess data
    processed_path = os.path.join(DATA_DIR, f"processed_dataset_{exp_name}.pt")
    
    metadata = preprocess_and_export(
        data_dir=DATA_DIR,
        keyboard_file=KEYBOARD_FILE,
        left_file=LEFT_FILE,
        right_file=RIGHT_FILE,
        output_path=processed_path,
        max_seq_length=config['max_seq_length'],
        normalize=True,
        apply_filtering=True
    )
    
    # Load dataset
    dataset = load_preprocessed_dataset(processed_path)
    print(f"Dataset: {len(dataset)} samples | Input dim: {dataset.input_dim}")
    
    # Create model
    model = create_model_from_dataset(
        dataset,
        model_type=config['model_type'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        bidirectional=config['bidirectional'],
        dropout=config['dropout'],
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Train
    trainer = GIKTrainer(
        model=model,
        dataset=dataset,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        device=DEVICE
    )
    
    history = trainer.train(
        epochs=config['epochs'],
        early_stopping_patience=config['early_stopping'],
        save_path=f'best_model_{exp_name}.pt'
    )
    
    # Get final metrics
    final_train_acc = max(history['train_acc'])
    final_val_acc = max(history['val_acc'])
    best_val_loss = min(history['val_loss'])
    epochs_trained = len(history['train_loss'])
    
    # Test set evaluation
    trainer.load_best_model(f'best_model_{exp_name}.pt')
    test_loss, test_acc = trainer.evaluate_test()
    
    result = {
        'name': exp_name,
        'config': config,
        'num_params': num_params,
        'best_train_acc': final_train_acc,
        'best_val_acc': final_val_acc,
        'best_val_loss': best_val_loss,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'epochs_trained': epochs_trained,
    }
    
    print(f"\nRESULTS for {exp_name}:")
    print(f"  Best Train Acc: {final_train_acc:.4f}")
    print(f"  Best Val Acc: {final_val_acc:.4f}")
    print(f"  Test Acc: {test_acc:.4f}")
    print(f"  Epochs: {epochs_trained}")
    
    # Cleanup temp files
    if os.path.exists(processed_path):
        os.remove(processed_path)
    metadata_path = processed_path.replace('.pt', '_metadata.json')
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    
    return result


def main():
    print(f"GIK Model Experiments")
    print(f"Device: {DEVICE}")
    print(f"Testing {len(EXPERIMENTS)} configurations")
    
    results = []
    
    for exp in EXPERIMENTS:
        try:
            result = run_experiment(exp['name'], exp['config'])
            results.append(result)
        except Exception as e:
            print(f"ERROR in {exp['name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': exp['name'],
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    # Sort by validation accuracy
    valid_results = [r for r in results if 'error' not in r]
    valid_results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    print(f"\n{'Name':<25} {'Model':<12} {'Hidden':<8} {'Val Acc':<10} {'Test Acc':<10} {'Params':<12}")
    print("-" * 80)
    
    for r in valid_results:
        print(f"{r['name']:<25} {r['config']['model_type']:<12} {r['config']['hidden_dim']:<8} "
              f"{r['best_val_acc']:.4f}     {r['test_acc']:.4f}     {r['num_params']:,}")
    
    if valid_results:
        best = valid_results[0]
        print(f"\n{'='*80}")
        print(f"BEST CONFIGURATION: {best['name']}")
        print(f"{'='*80}")
        print(f"Validation Accuracy: {best['best_val_acc']:.4f}")
        print(f"Test Accuracy: {best['test_acc']:.4f}")
        print(f"\nRecommended CONFIG for train_model.ipynb:")
        print(json.dumps(best['config'], indent=4))
    
    # Save results
    results_file = os.path.join(DATA_DIR, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'device': DEVICE,
            'results': results,
            'best': best['name'] if valid_results else None
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
