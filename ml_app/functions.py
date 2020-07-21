import os
from tensorflow.keras.models import  load_model


MODEL_PATH = 'saved'

load_model('saved_model/my_model')

def detect_using_xray(file):
    try:
        # hypothetical function that on call with the file (xray) detect and returns the
        # status of the xray
        pass
    except:
        pass