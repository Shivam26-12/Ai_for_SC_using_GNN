import os
import subprocess
import pandas as pd

def main():
    stores = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    all_subs = []
    
    # Set to utf-8 encoding to prevent Windows emoji errors
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print("🚀 Starting Full M5 Pipeline Runner (10 Stores)")
    print("This sequence will launch sequential 50-epoch GNN trainings to respect GPU VRAM limits.")

    for i, store in enumerate(stores):
        print(f"\n========================================================")
        print(f"[{i+1}/10] 🚂 Training M5 SigGNN for store: {store}")
        print(f"========================================================")
        
        # Run the full integration pipeline for this specific store
        cmd = ["python", "run_m5.py", "--store", store, "--epochs", "50"]
        result = subprocess.run(cmd, env=env)
        
        if result.returncode != 0:
            print(f"\n❌ Error occurred while training {store}. Halting execution...")
            break
            
        # If successful, load its isolated submission CSV and append it
        if os.path.exists('submission.csv'):
            df = pd.read_csv('submission.csv')
            all_subs.append(df)
            print(f"✅ Success: Extracted {len(df)} predictions for {store}.")
        else:
            print(f"⚠️ Warning: 'submission.csv' was missing after running {store}.")

    # Merge all individual DataFrames vertically
    if len(all_subs) > 0:
        final_sub = pd.concat(all_subs, axis=0)
        final_sub.to_csv('submission_full_m5.csv', index=False)
        
        print("\n========================================================")
        print(f"🎉 EXECUTION COMPLETE: Merged {len(all_subs)} stores.")
        print(f"   Final CSV Shape: {final_sub.shape} (Expected: 60980 rows)")
        print(f"   Saved to: ./submission_full_m5.csv")
        print("   Upload 'submission_full_m5.csv' directly to Kaggle!")
        print("========================================================")

if __name__ == '__main__':
    main()
