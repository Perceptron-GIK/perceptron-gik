#!/usr/bin/env python3
"""
GIK Model Experiments V2 - Comprehensive experiments with larger dataset (data_hazel_2).

Run specific experiment: python run_experiments_v2.py --experiment <name>
Run all experiments: python run_experiments_v2.py --all
List experiments: python run_experiments_v2.py --list
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
import numpy as np
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pretraining import preprocess_multiple_sources, load_preprocessed_dataset
from ml.models.basic_nn import create_model_from_dataset, GIKTrainer

torch.manual_seed(42)
np.random.seed(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Data paths - Using data_hazel_2 with multiple sources
DATA_DIR = "data_hazel_2"
KEYBOARD_FILES = ["Keyboard_1.csv", "Keyboard_2.csv", "Keyboard_3.csv", "Keyboard_4.csv"]
LEFT_FILES = ["Left_1.csv", "Left_2.csv", "Left_3.csv", "Left_4.csv"]
RIGHT_FILES = None

# Results directory
RESULTS_DIR = "experiment_results_v2"

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

EXPERIMENTS = {
    # Baseline LSTM (current best)
    'lstm_baseline': {
        'description': 'Baseline LSTM with focal loss (current best)',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.5,
            'batch_size': 32,
            'learning_rate': 5e-4,
            'weight_decay': 1e-3,
            'epochs': 100,
            'early_stopping': 20,
            'model_type': 'lstm',
            'use_focal_loss': True,
            'focal_gamma': 2.0,
        }
    },
    
    # Transformer variants
    'transformer_small': {
        'description': 'Small Transformer (4 heads, 2 layers)',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 64,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.3,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'epochs': 100,
            'early_stopping': 25,
            'model_type': 'transformer',
            'use_focal_loss': True,
            'focal_gamma': 2.0,
        }
    },
    'transformer_medium': {
        'description': 'Medium Transformer (8 heads, 4 layers)',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 128,
            'num_layers': 4,
            'num_heads': 8,
            'dropout': 0.3,
            'batch_size': 32,
            'learning_rate': 5e-5,
            'weight_decay': 1e-4,
            'epochs': 100,
            'early_stopping': 25,
            'model_type': 'transformer',
            'use_focal_loss': True,
            'focal_gamma': 2.0,
        }
    },
    'transformer_large': {
        'description': 'Large Transformer (8 heads, 6 layers, 256 dim)',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.4,
            'batch_size': 16,
            'learning_rate': 1e-5,
            'weight_decay': 1e-4,
            'epochs': 100,
            'early_stopping': 30,
            'model_type': 'transformer',
            'use_focal_loss': True,
            'focal_gamma': 2.0,
        }
    },
    
    # Attention LSTM variants
    'attention_lstm_small': {
        'description': 'Attention LSTM (64 hidden, 2 layers)',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 64,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.4,
            'batch_size': 32,
            'learning_rate': 5e-4,
            'weight_decay': 1e-3,
            'epochs': 100,
            'early_stopping': 20,
            'model_type': 'attention_lstm',
            'use_focal_loss': True,
            'focal_gamma': 2.0,
        }
    },
    'attention_lstm_large': {
        'description': 'Attention LSTM (256 hidden, 3 layers)',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 256,
            'num_layers': 3,
            'bidirectional': True,
            'dropout': 0.5,
            'batch_size': 16,
            'learning_rate': 3e-4,
            'weight_decay': 1e-3,
            'epochs': 100,
            'early_stopping': 25,
            'model_type': 'attention_lstm',
            'use_focal_loss': True,
            'focal_gamma': 2.0,
        }
    },
    
    # LSTM variants with different configs
    'lstm_large': {
        'description': 'Large LSTM (256 hidden, 3 layers)',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 256,
            'num_layers': 3,
            'bidirectional': True,
            'dropout': 0.5,
            'batch_size': 16,
            'learning_rate': 3e-4,
            'weight_decay': 1e-3,
            'epochs': 100,
            'early_stopping': 25,
            'model_type': 'lstm',
            'use_focal_loss': True,
            'focal_gamma': 2.0,
        }
    },
    'lstm_no_focal': {
        'description': 'LSTM without focal loss (cross entropy)',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.5,
            'batch_size': 32,
            'learning_rate': 5e-4,
            'weight_decay': 1e-3,
            'epochs': 100,
            'early_stopping': 20,
            'model_type': 'lstm',
            'use_focal_loss': False,
            'focal_gamma': 2.0,
        }
    },
    
    # GRU variants
    'gru_baseline': {
        'description': 'Bidirectional GRU (128 hidden)',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.5,
            'batch_size': 32,
            'learning_rate': 5e-4,
            'weight_decay': 1e-3,
            'epochs': 100,
            'early_stopping': 20,
            'model_type': 'gru',
            'use_focal_loss': True,
            'focal_gamma': 2.0,
        }
    },
    'gru_large': {
        'description': 'Large GRU (256 hidden, 3 layers)',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 256,
            'num_layers': 3,
            'bidirectional': True,
            'dropout': 0.5,
            'batch_size': 16,
            'learning_rate': 3e-4,
            'weight_decay': 1e-3,
            'epochs': 100,
            'early_stopping': 25,
            'model_type': 'gru',
            'use_focal_loss': True,
            'focal_gamma': 2.0,
        }
    },
    
    # Different sequence lengths
    'lstm_seq50': {
        'description': 'LSTM with shorter sequence (50)',
        'config': {
            'max_seq_length': 50,
            'hidden_dim': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.5,
            'batch_size': 32,
            'learning_rate': 5e-4,
            'weight_decay': 1e-3,
            'epochs': 100,
            'early_stopping': 20,
            'model_type': 'lstm',
            'use_focal_loss': True,
            'focal_gamma': 2.0,
        }
    },
    'lstm_seq200': {
        'description': 'LSTM with longer sequence (200)',
        'config': {
            'max_seq_length': 200,
            'hidden_dim': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.5,
            'batch_size': 16,
            'learning_rate': 5e-4,
            'weight_decay': 1e-3,
            'epochs': 100,
            'early_stopping': 20,
            'model_type': 'lstm',
            'use_focal_loss': True,
            'focal_gamma': 2.0,
        }
    },
    
    # Different focal loss gamma
    'lstm_focal_gamma1': {
        'description': 'LSTM with focal loss gamma=1.0',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.5,
            'batch_size': 32,
            'learning_rate': 5e-4,
            'weight_decay': 1e-3,
            'epochs': 100,
            'early_stopping': 20,
            'model_type': 'lstm',
            'use_focal_loss': True,
            'focal_gamma': 1.0,
        }
    },
    'lstm_focal_gamma3': {
        'description': 'LSTM with focal loss gamma=3.0',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.5,
            'batch_size': 32,
            'learning_rate': 5e-4,
            'weight_decay': 1e-3,
            'epochs': 100,
            'early_stopping': 20,
            'model_type': 'lstm',
            'use_focal_loss': True,
            'focal_gamma': 3.0,
        }
    },
    
    # CNN baseline
    'cnn_baseline': {
        'description': 'CNN with temporal convolutions',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.4,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-3,
            'epochs': 100,
            'early_stopping': 20,
            'model_type': 'cnn',
            'use_focal_loss': True,
            'focal_gamma': 2.0,
        }
    },
    
    # Higher learning rates
    'lstm_lr_high': {
        'description': 'LSTM with higher learning rate (1e-3)',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.5,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-3,
            'epochs': 100,
            'early_stopping': 20,
            'model_type': 'lstm',
            'use_focal_loss': True,
            'focal_gamma': 2.0,
        }
    },
    
    # Lower dropout
    'lstm_dropout_low': {
        'description': 'LSTM with lower dropout (0.3)',
        'config': {
            'max_seq_length': 100,
            'hidden_dim': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.3,
            'batch_size': 32,
            'learning_rate': 5e-4,
            'weight_decay': 1e-3,
            'epochs': 100,
            'early_stopping': 20,
            'model_type': 'lstm',
            'use_focal_loss': True,
            'focal_gamma': 2.0,
        }
    },
}


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def run_experiment(exp_name: str, config: dict, description: str) -> dict:
    """Run a single experiment and return results."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Description: {description}")
    print(f"{'='*70}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Preprocess data with multiple sources
    processed_path = os.path.join(DATA_DIR, f"processed_{exp_name}.pt")
    
    print(f"\nPreprocessing data from {len(KEYBOARD_FILES)} sources...")
    metadata = preprocess_multiple_sources(
        data_dir=DATA_DIR,
        keyboard_files=KEYBOARD_FILES,
        left_files=LEFT_FILES,
        right_files=RIGHT_FILES,
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
        dropout=config['dropout'],
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model type: {config['model_type']}")
    
    # Train
    trainer = GIKTrainer(
        model=model,
        dataset=dataset,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0),
        device=DEVICE,
        use_focal_loss=config.get('use_focal_loss', False),
        focal_gamma=config.get('focal_gamma', 2.0),
    )
    
    model_save_path = os.path.join(RESULTS_DIR, f'best_model_{exp_name}.pt')
    history = trainer.train(
        epochs=config['epochs'],
        early_stopping_patience=config['early_stopping'],
        save_path=model_save_path
    )
    
    # Get final metrics
    best_train_acc = max(history['train_acc'])
    best_val_acc = max(history['val_acc'])
    best_val_loss = min(history['val_loss'])
    epochs_trained = len(history['train_loss'])
    
    # Test set evaluation
    trainer.load_best_model(model_save_path)
    test_loss, test_acc = trainer.evaluate_test()
    
    result = {
        'name': exp_name,
        'description': description,
        'config': config,
        'num_params': num_params,
        'num_samples': len(dataset),
        'best_train_acc': float(best_train_acc),
        'best_val_acc': float(best_val_acc),
        'best_val_loss': float(best_val_loss),
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
        'epochs_trained': epochs_trained,
        'device': DEVICE,
        'timestamp': datetime.now().isoformat(),
    }
    
    print(f"\n{'='*50}")
    print(f"RESULTS for {exp_name}:")
    print(f"{'='*50}")
    print(f"  Best Train Acc: {best_train_acc:.4f}")
    print(f"  Best Val Acc:   {best_val_acc:.4f}")
    print(f"  Test Acc:       {test_acc:.4f}")
    print(f"  Epochs:         {epochs_trained}")
    print(f"  Parameters:     {num_params:,}")
    
    # Save individual result
    result_file = os.path.join(RESULTS_DIR, f'result_{exp_name}.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {result_file}")
    
    # Cleanup preprocessed data
    if os.path.exists(processed_path):
        os.remove(processed_path)
    metadata_path = processed_path.replace('.pt', '_metadata.json')
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    
    return result


def list_experiments():
    """List all available experiments."""
    print("\nAvailable Experiments:")
    print("=" * 70)
    for name, exp in EXPERIMENTS.items():
        model = exp['config']['model_type']
        hidden = exp['config']['hidden_dim']
        layers = exp['config']['num_layers']
        print(f"  {name:<25} | {model:<15} | h={hidden:<4} l={layers} | {exp['description']}")
    print("=" * 70)
    print(f"\nTotal: {len(EXPERIMENTS)} experiments")


def summarize_results():
    """Summarize all experiment results."""
    ensure_results_dir()
    results = []
    
    for filename in os.listdir(RESULTS_DIR):
        if filename.startswith('result_') and filename.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, filename)) as f:
                results.append(json.load(f))
    
    if not results:
        print("No results found.")
        return
    
    # Sort by test accuracy
    results.sort(key=lambda x: x.get('test_acc', 0), reverse=True)
    
    print("\n" + "=" * 90)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 90)
    print(f"\n{'Name':<25} {'Model':<15} {'Hidden':<8} {'Val Acc':<10} {'Test Acc':<10} {'Params':<12}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['name']:<25} {r['config']['model_type']:<15} {r['config']['hidden_dim']:<8} "
              f"{r['best_val_acc']:.4f}     {r['test_acc']:.4f}     {r['num_params']:,}")
    
    if results:
        best = results[0]
        print(f"\n{'='*90}")
        print(f"BEST CONFIGURATION: {best['name']}")
        print(f"{'='*90}")
        print(f"Description: {best['description']}")
        print(f"Validation Accuracy: {best['best_val_acc']:.4f}")
        print(f"Test Accuracy: {best['test_acc']:.4f}")
        print(f"\nRecommended CONFIG:")
        print(json.dumps(best['config'], indent=4))


def main():
    parser = argparse.ArgumentParser(description='GIK Model Experiments V2')
    parser.add_argument('--experiment', '-e', type=str, help='Run specific experiment by name')
    parser.add_argument('--all', '-a', action='store_true', help='Run all experiments')
    parser.add_argument('--list', '-l', action='store_true', help='List available experiments')
    parser.add_argument('--summary', '-s', action='store_true', help='Summarize results')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    if args.summary:
        summarize_results()
        return
    
    ensure_results_dir()
    print(f"GIK Model Experiments V2")
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_DIR} ({len(KEYBOARD_FILES)} sources)")
    
    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            list_experiments()
            return
        exp = EXPERIMENTS[args.experiment]
        run_experiment(args.experiment, exp['config'], exp['description'])
    elif args.all:
        print(f"Running all {len(EXPERIMENTS)} experiments...")
        for name, exp in EXPERIMENTS.items():
            try:
                run_experiment(name, exp['config'], exp['description'])
            except Exception as e:
                print(f"ERROR in {name}: {e}")
                import traceback
                traceback.print_exc()
        summarize_results()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python run_experiments_v2.py --list")
        print("  python run_experiments_v2.py --experiment transformer_medium")
        print("  python run_experiments_v2.py --all")
        print("  python run_experiments_v2.py --summary")


if __name__ == "__main__":
    main()
