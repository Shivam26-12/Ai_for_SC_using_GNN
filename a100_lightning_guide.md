# Comprehensive Guide: Migrating SigGNN to Lightning AI (A100)

This guide takes you step-by-step from your local machine to running your A100-optimized SigGNN model on the Lightning AI cloud platform.

## Step 1: Provisioning the Studio

1. Log into your [Lightning AI](https://lightning.ai/) account.
2. Click **New Studio** (you can start with a blank PyTorch studio).
3. Once inside, look at the top-right corner to change your compute hardware.
4. Scale up the hardware by selecting **A100 (40GB)** or **A100 (80GB)**. _(Note: Make sure the instance you choose has at least 16 CPUs and 64GB+ of system memory; otherwise, the CPU dataloading bottleneck will slow down your multi-store execution)_.

> [!TIP]
> A100 instances charge by the hour. Do your preliminary reading/setup on the free lightweight CPU tier, and only click the "Switch to A100" button right before you are ready to execute the training script!

## Step 2: Uploading the Project

You do not need to upload everything. It is strongly advised **not** to upload the massive dataset from your local PC's Wi-Fi. 
Upload the codebase by dragging and dropping your `m5_siggnn` folder into the Lightning Studio file tree, or by linking it to your GitHub repository and cloning it via the terminal.

## Step 3: Installing Dependencies & Kaggle API

Open a terminal inside the Lightning AI Studio and run:

```bash
# Install the exact requirements 
pip install -r requirements.txt

# Install Kaggle to easily fetch the M5 data straight to the server
pip install kaggle
```

## Step 4: Secure the M5 Dataset

Because server-to-server downloads occur at over 1+ Gbps, fetching the M5 dataset directly from Kaggle takes less than 2 minutes.

1. Go to your Kaggle Account Settings on your local browser and click "Create New API Token". This downloads `kaggle.json`.
2. Upload `kaggle.json` into your Lightning AI studio.
3. Move the token and set permissions in the studio terminal:
```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
4. Download and extract the dataset into your project folder:
```bash
cd m5_siggnn
mkdir -p dataset
cd dataset
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip
```
5. *(Optional)* Delete the zip file `rm m5-forecasting-accuracy.zip` to save disk space.

## Step 5: Executing the Workload

Now that your data is loaded and your environment is primed, execute the A100 configurations we built.

> [!CAUTION]
> The data-preparation step takes around **30-45 minutes of heavy CPU work** before the A100 takes over. It's highly recommended to use `tmux` or `nohup` so that if your browser tabs close, the pipeline continues running in the cloud!

### Option A: Chaos Engineering & Full Evaluation
If you want to run the full diagnostic suite with the Hawkes processes and resilience metrics using the newly scaled model:
```bash
python main.py --mode a100
```

### Option B: The Kaggle Submission Run
If you want to bypass the chaos testing and heavily optimize raw WRMSSE across all 10 M5 stores at once, outputting `submission.csv`:
```bash
python run_m5.py --store all --a100
```
_If `run_m5.py` starts allocating too much CPU RAM during the 10-store dataloader process depending on your exact Lightning instance size, you can always revert to sequential loops (`--store CA_1`, `--store CA_2`, etc.)._

## Step 6: Offboarding

1. Right-click the generated `training_results.txt` and `submission.csv` files and download them to your local PC.
2. If you saved checkpoints, download them from the `checkpoints/` directory.
3. **CRITICAL:** Go to the top right of the Studio and click **Stop Studio**. A100 GPUs are expensive and running them idle over the weekend will drain your cloud credits.
