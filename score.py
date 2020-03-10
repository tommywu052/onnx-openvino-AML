# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.


import sys
import time
import io
import csv


# Imports for inferencing
import onnxruntime as rt
from inference import run_onnx
import numpy as np
import cv2

# Imports for communication w/IOT Hub
from iothub_client import IoTHubModuleClient, IoTHubClientError, IoTHubTransportProvider
from iothub_client import IoTHubMessage, IoTHubMessageDispositionResult, IoTHubError
from azureml.core.model import Model

# Imports for the http server
from flask import Flask, request
import json

# Imports for storage
import os
# from azure.storage.blob import BlockBlobService, PublicAccess, AppendBlobService
import random
import string
import csv
from datetime import datetime
from pytz import timezone  
import time
import json

class HubManager(object):
    def __init__(
            self,
            protocol=IoTHubTransportProvider.MQTT):
        self.client_protocol = protocol
        self.client = IoTHubModuleClient()
        self.client.create_from_environment(protocol)

        # set the time until a message times out
        self.client.set_option("messageTimeout", MESSAGE_TIMEOUT)

    # Forwards the message received onto the next stage in the process.
    def forward_event_to_output(self, outputQueueName, event, send_context):
        self.client.send_event_async(
            outputQueueName, event, send_confirmation_callback, send_context)



def send_confirmation_callback(message, result, user_context):
    """
    Callback received when the message that we're forwarding is processed.
    """
    print("Confirmation[%d] received for message with result = %s" % (user_context, result))


def get_tinyyolo_frame_from_encode(msg):
    """
    Formats jpeg encoded msg to frame that can be processed by tiny_yolov2
    """
    #inp = np.array(msg).reshape((len(msg),1))
    #frame = cv2.imdecode(inp.astype(np.uint8), 1)
    frame = cv2.cvtColor(msg, cv2.COLOR_BGR2RGB)
    
    # resize and pad to keep input frame aspect ratio
    h, w = frame.shape[:2]
    tw = 416 if w > h else int(np.round(416.0 * w / h))
    th = 416 if h > w else int(np.round(416.0 * h / w))
    frame = cv2.resize(frame, (tw, th))
    pad_value=114
    top = int(max(0, np.round((416.0 - th) / 2)))
    left = int(max(0, np.round((416.0 - tw) / 2)))
    bottom = 416 - top - th
    right = 416 - left - tw
    frame = cv2.copyMakeBorder(frame, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])
    
    frame = np.ascontiguousarray(np.array(frame, dtype=np.float32).transpose(2, 0, 1)) # HWC -> CHW
    frame = np.expand_dims(frame, axis=0)
    return frame

def run(msg):
    # this is a dummy function required to satisfy AML-SDK requirements.
    return msg

def init():
    # Choose HTTP, AMQP or MQTT as transport protocol.  Currently only MQTT is supported.
    PROTOCOL = IoTHubTransportProvider.MQTT
    DEVICE = 0 # when device is /dev/video0
    LABEL_FILE = "labels.txt"
    MODEL_FILE = "Model.onnx"
    global MESSAGE_TIMEOUT
    MESSAGE_TIMEOUT = 1000

    
    # Create the IoT Hub Manager to send message to IoT Hub
    print("trying to make IOT Hub manager")
    
    hub_manager = HubManager(PROTOCOL)

    if not hub_manager:
        print("Took too long to make hub_manager, exiting program.")
        print("Try restarting IotEdge or this module.")
        sys.exit(1)

    # Get Labels from labels file 
    labels_file = open(LABEL_FILE)
    labels_string = labels_file.read()
    labels = labels_string.split(",")
    labels_file.close()
    label_lookup = {}
    for i, val in enumerate(labels):
        label_lookup[val] = i

    # get model path from within the container image
    model_path=Model.get_model_path(MODEL_FILE)
    
    # Loading ONNX model
    sys.path.append("/opt/intel/openvino_2019.1.144/deployment_tools/model_optimizer")
    print('sys.path = {}' .format(sys.path))

    print("loading model to ONNX Runtime...")
    start_time = time.time()
    ort_session = rt.InferenceSession(model_path)
    print("loaded after", time.time()-start_time,"s")

    # start reading frames from video endpoint
    
    cap = cv2.VideoCapture(DEVICE)

    while cap.isOpened():
        _, _ = cap.read()
        ret, img_frame = cap.read()       
        if not ret:
            print('no video RESETTING FRAMES TO 0 TO RUN IN LOOP')
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        """ 
        Handles incoming inference calls for each fames. Gets frame from request and calls inferencing function on frame.
        Sends result to IOT Hub.
        """
        try:
                        
            draw_frame = img_frame
            start_time = time.time()
            # pre-process the frame to flatten, scale for tiny-yolo
            #ret, img_frame = cv2.imencode('.jpg', img_frame)
            #img_frame = img_frame.flatten().tolist()
            frame = get_tinyyolo_frame_from_encode(img_frame)
            
            # run the inference session for the given input frame
            objects = run_onnx(frame, ort_session, draw_frame)
            
            # LOOK AT OBJECTS AND CHECK PREVIOUS STATUS TO APPEND
            num_objects = len(objects) 
            print("NUMBER OBJECTS DETECTED:", num_objects)                               
            print("PROCESSED IN:",time.time()-start_time,"s")            
            if num_objects > 0:
                output_IOT = IoTHubMessage(json.dumps(objects))
                hub_manager.forward_event_to_output("inferenceoutput", output_IOT, 0)
            continue
        except Exception as e:
            print('EXCEPTION:', str(e))
            continue
