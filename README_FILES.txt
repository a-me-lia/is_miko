src/
│
├── data_preprocessing/
│   ├── video_preprocessing.py │ methods to process video into frames with bodies and faces with cv2
│
├── model_training/
│   ├── model.py │ Tensorflow model with the create_model method
│   ├── train.py │ Contains the scripture for n-1 step preprocessing, data augmentation and invoking model.py
│
├── prediction/
│   ├── predict.py │ Uses model trained with train.py to make predictions
│
├── gui/
│   ├── dataset_gui.py │ Currently has all dataset CSV handling methods and creates the GUI*
│   ├── train_gui.py │ GUI for training, includes video preprocessing*
│   ├── predict_gui.py │ GUI for making prediction*
│
├── main.py │ Runs the GUI components*
└── setEnv.py │ File for the TF training device utilization variables

* may be subsitituted with "CLI" for CLI version


Other files/folders:

Included
rsc... │ Used by cv2 for detection. 


Generated:
videoData │ Default folder selected videos is moved to
processedData │ Folder containing folders with grabbed frames from each video after preprocessing
checkpoints │ Folder containing mid-training .veras model saves