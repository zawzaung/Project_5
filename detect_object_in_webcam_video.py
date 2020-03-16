#!/usr/bin/env python
# coding: utf-8

# # Detection Objects in a webcam image stream

# <table align="left"><td>
#   <a target="_blank"  href="https://colab.research.google.com/github/TannerGilbert/Tutorials/blob/master/Tensorflow%20Object%20Detection/detect_object_in_webcam_video.ipynb">
#     <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab
#   </a>
# </td><td>
#   <a target="_blank"  href="https://github.com/TannerGilbert/Tutorials/blob/master/Tensorflow%20Object%20Detection/detect_object_in_webcam_video.ipynb">
#     <img width=32px src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
# </td></table>

# This notebook is based on the [official Tensorflow Object Detection demo](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb) and only contains some slight changes. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2


from IPython.display import clear_output


cap = cv2.VideoCapture(0)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


# In[5]:





# ## Env setup

# In[2]:


# This is needed to display the images.




# ## Object detection imports
# Here are the imports from the object detection module.

# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[3]:


import pathlib


# In[4]:


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# In[5]:


MODEL_NAME = 'inference_graph'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('training','labelmap.pbtxt')
NUM_CLASSES = 2


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:


label_map= label_map_util.load_labelmap(PATH_TO_LABELS)
categories=label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# # Detection

# In[ ]:


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            
            ret, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np,axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes,scores,classes,num_detections)=sess.run(
                [boxes,scores,classes,num_detections],
                feed_dict={image_tensor:image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            
            plt.imshow(image_np)
            plt.axis('off')
            plt.show()
            clear_output(wait=True)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break


# In[ ]:




