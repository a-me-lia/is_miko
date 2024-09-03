
# is_miko

Purpose: To accurately predict if an image contains Yae Miko from the game Genshin Impact.
Releases are not yet packaged for this project

---
## Usage

**VSCODE:** Open this directory in thw workspace, and use CTRLP to install requirements.txt and initialize the .venv.

**Requirements:** 

- Training requires at least 20G of free VRAM for 224x224x3 RGB images and a batch size of 216 (GPU). 
- Only possible to run on Windows or systems with CUDA support, requires CUDA 12.4 and Python 3.10


**To run:** & c:/<PATH_TO_PROJECT_>/.venv/Scripts/python.exe c:/<PATH_TO_PROJECT_>/src/main.py or click the run button with src/main.py open in the editor to run

---

## File structure

The file structure is located in README-FILES.txt

---

## Performance

Currently the model severely over-fits. It is able to achieve 100% PASS on training dataset, augmented, but has difficulty detecting true positives in new datasets, or datasets with slightly different style such as cosplay, or different artstyle MIKO photos. 

See the results in miko27.png, the result on prediction for the entire dataset after 27 epochs of 216 batch learning rate 0.00002 training. 

The same overfitting issue is present for other datasets, such as a binary dataset consisting of Rem and Ganyu images.