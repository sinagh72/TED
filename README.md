## Local Learning Phase

### 1. Data Preparation
1. Obtain the dataset (`QIAI-FL.zip`) and extract it:
   - Right-click the zip file and select **Extract**.
2. Create three CSV files named `train.csv`, `val.csv`, and `test.csv` with the following columns:
   - **Directory**: Path to the image files.
   - **Label**: Associated label for each image.
3. Place these CSV files in the `data` directory.
   - Example CSV content:
     ```
     Directory,Label
     path/to/image1.jpg,label1
     path/to/image2.jpg,label2
     ```

### 2. Install Python
1. Download Python version **3.12.7** from the [Python website](https://www.python.org/downloads/release/python-3127/).
2. Install Python:
   - Right-click on the installer and select **Run as Administrator**.
   - Check the box **Add Python to PATH** and click **Install Now**.
3. Verify installation:
   - Open Command Prompt (CMD) and type: `python --version`.

### 3. Create a Python Virtual Environment
1. Open Command Prompt:
   - Press `Win + R`, type `cmd`, and hit Enter.
2. Navigate to the project directory:
   - Use `cd path_to_project_folder`.
3. Create and activate the virtual environment:
   - Run: `python -m venv qiai-lab`.
   - Activate it: `qiai-lab\Scripts\activate`.
   - You should see `(qiai-lab)` at the beginning of the prompt.
4. Install required libraries:
   - Install PyTorch: 
     ```
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
     ```
   - Install additional dependencies:
     ```
     pip install -r requirements.txt
     ```

### 4. Run the Program
1. Navigate to the project folder using the `cd` command.
2. Activate the virtual environment:
   - Run: `qiai-lab\Scripts\activate`.
3. Start training:
   - Execute: `python train_local.py`.

---

## Pre-Training Phase

### 1. Dataset Preparation
1. Download the dataset and unzip it into the `/data` folder.

### 2. Install Additional Python Libraries
1. Ensure that all required libraries listed in `requirements.txt` are installed:
   - Run: `pip install -r requirements.txt`.

### 3. Download the Pre-Trained SimMIM Model
1. Download the pre-trained SimMIM model from the model hub.
2. Place the model file in the `models` directory.

### 4. Start Pre-Training
1. Use the following command to begin pre-training:
   ```
   torchrun --nproc_per_node=1 pretrain_simmim.py \
   --cfg ./config/simmim_pretrain__swinv2_base__img192_window12__800ep.yaml \
   --resume ./models/swinv2_base_patch4_window12_192_22k.pth \
   --batch-size 128
   ```
   - Replace `--nproc_per_node` with the number of GPUs to use.

---

### Notes
- Ensure that all files and folders (`data`, `models`, `requirements.txt`, etc.) are properly organized as described.
- Follow the steps in the exact order to ensure smooth execution.
- For additional information or troubleshooting, refer to the provided PDF or project documentation.

--- 

This **README.md** ensures clarity and comprehensiveness for both local training and pre-training tasks. Let me know if you need adjustments!