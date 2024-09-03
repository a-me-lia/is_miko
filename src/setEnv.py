import os

def set_env():
    os.environ['TF_NUM_INTEROP_THREADS'] = '16'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '16'
#   os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'