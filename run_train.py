#!/usr/bin/env python
"""
SigGNN Training Entry Point.
Simplified run script for starting M5 training on local GPU.
"""
import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Start SigGNN Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="./dataset", 
                        help="Path to M5 dataset directory")
    parser.add_argument("--resume", type=str, default="", 
                        help="Path to checkpoint to resume from (e.g. ./checkpoints/best_model.pt)")
    parser.add_argument("--no-hawkes", action="store_true", 
                        help="Disable Hawkes chaos (run faster, but less realistic)")
    parser.add_argument("--debug", action="store_true", 
                        help="Run in debug mode (fast subset of data)")
    
    args = parser.parse_args()
    
    # Pre-flight check
    if not os.path.exists(args.data_dir):
        print(f"⚠️ Warning: Dataset directory '{args.data_dir}' not found.")
        print("Please download the M5 Forecasting Accuracy dataset from Kaggle")
        print("and extract it here, or specify the correct path using --data-dir.")
        print("Expected files: sales_train_evaluation.csv, calendar.csv, sell_prices.csv")
        response = input("\nContinue anyway? (It will fail unless files exist) [y/N]: ")
        if response.lower() != 'y':
            sys.exit(1)

    # Build command
    cmd = [sys.executable, "main.py"]
    
    cmd.extend(["--data-dir", args.data_dir])
    
    if args.debug:
        cmd.extend(["--mode", "debug"])
    else:
        cmd.extend(["--mode", "full"])
        
    if args.resume:
        cmd.extend(["--resume", args.resume])
        
    if args.no_hawkes:
        cmd.append("--no-hawkes")

    print(f"\n🚀 Running Command: {' '.join(cmd)}")
    print(f"============================================================")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed with exit code {e.returncode}.")

if __name__ == "__main__":
    main()
