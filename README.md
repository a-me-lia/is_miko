
# is_miko

Purpose: To accurately predict if an image contains Yae Miko from the game Genshin Impact. This is a specific application of exploring how to best train machine learning models with a contrained/nonexistant dataset.

In this program, the user can create a dataset from scratch with classes by adding video paths to be coped in the dataset manager. When preprocessing is run, the video will be sampled every 24th frame and kept if HaarCascade detects faces or bodies. 

When training is run, there is option to adjust the training hyperparameters within the program. The training program itself does additional iage data preprocessing by applying augmentation and scaling to the desired 224/224/3 input size for resnet50. After each epoch a snapshot is created too. These snapshots are 100MB each, so please be mindful of running thousands of epochs and a storage-starved device.

After training is complete, the model file is spat out along with plots of the loss and validation numbers every epoch.

The prediction manager can select a folder to run batch predict and validate results, or run inferences one by one.

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