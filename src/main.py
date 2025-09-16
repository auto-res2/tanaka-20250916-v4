#!/usr/bin/env python3
"""
H2D-GAT Main Execution Script
Supports --smoke-test and --full-experiment flags
"""

import argparse
import yaml
import os
import sys
import json
from pathlib import Path

from .train import H2DGATTrainer
from .evaluate import H2DGATEvaluator
from .preprocess import DataPreprocessor

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_output_dirs():
    """Create required output directories"""
    os.makedirs('.research/iteration2/images', exist_ok=True)
    os.makedirs('.research/iteration2', exist_ok=True)

def run_experiment(config, experiment_type):
    """Run H2D-GAT experiment with given configuration"""
    print(f"=== H2D-GAT {experiment_type.upper()} EXPERIMENT ===")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Epochs: {config['experiment']['epochs']}")
    print(f"Model: {config['model']['num_layers']} layers, {config['model']['num_heads']} heads")
    print(f"Dataset: {config['dataset']['name']}")
    
    preprocessor = DataPreprocessor(config)
    data = preprocessor.load_and_preprocess()
    
    trainer = H2DGATTrainer(config)
    model, training_results = trainer.train(data)
    
    evaluator = H2DGATEvaluator(config)
    evaluation_results = evaluator.evaluate(model, data)
    
    results = {
        'experiment_type': experiment_type,
        'config': config,
        'training_results': training_results,
        'evaluation_results': evaluation_results
    }
    
    result_file = f'.research/iteration2/{experiment_type}_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n=== EXPERIMENT RESULTS ===")
    print(json.dumps(results, indent=2))
    print(f"\nResults saved to: {result_file}")
    print(f"Figures saved to: .research/iteration2/images/")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='H2D-GAT Experiment Runner')
    parser.add_argument('--smoke-test', action='store_true', 
                       help='Run smoke test with minimal configuration')
    parser.add_argument('--full-experiment', action='store_true',
                       help='Run full experiment with complete configuration')
    
    args = parser.parse_args()
    
    if not (args.smoke_test or args.full_experiment):
        parser.error('Must specify either --smoke-test or --full-experiment')
    
    ensure_output_dirs()
    
    if args.smoke_test:
        config = load_config('config/smoke_test.yaml')
        run_experiment(config, 'smoke_test')
    
    if args.full_experiment:
        config = load_config('config/full_experiment.yaml')
        run_experiment(config, 'full_experiment')

if __name__ == '__main__':
    main()
