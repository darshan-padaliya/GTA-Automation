import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import numpy as np
#import os
import pathlib
#import six.moves.urllib as urllib
#import sys
#import tarfile

#import zipfile

#from collections import defaultdict
#from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image
#from IPython.display import display





from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util





# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile





def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))

  return model





# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'C:\Darshan\GTA Automation\models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)





model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)





#print(detection_model.signatures['serving_default'].inputs)





#detection_model.signatures['serving_default'].output_dtypes





#detection_model.signatures['serving_default'].output_shapes


#


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(output_dict['detection_masks'], output_dict['detection_boxes'],image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


#


# def show_inference(model, image_np):
#   # the array based representation of the image will be used later in order to prepare the
#   # result image with boxes and labels on it.
#   # -----------------image_np = np.array(Image.open(image_path))---------------------#
#   # Actual detection.
#   output_dict = run_inference_for_single_image(model, image_np)
#   # Visualization of the results of a detection.
#   vis_util.visualize_boxes_and_labels_on_image_array(
#       image_np,
#       output_dict['detection_boxes'],
#       output_dict['detection_classes'],
#       output_dict['detection_scores'],
#       category_index,
#       instance_masks=output_dict.get('detection_masks_reframed', None),
#       use_normalized_coordinates=True,
#       line_thickness=8)

#   #----------------display(Image.fromarray(image_np))------------------# 
#   cv2.imshow('window',image_np)
# #   if cv2.waitKey(25) & 0xFF == ord('q'):
# #       cv2.destroyAllWindows()
# #       break  


#


# from grabscreen import grab_screen

# screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (800,450))
# image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

# show_inference(detection_model, image_np)


#


from grabscreen import grab_screen
import cv2
import time

last_time = time.time()
while True:
  screen = cv2.resize(grab_screen(region=(0,40,1000,700)), (800,450))
  image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

  output_dict = run_inference_for_single_image(detection_model, image_np)

  vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks_reframed', None),
    use_normalized_coordinates=True,
    line_thickness=8)

  cv2.imshow('window',image_np)
  if cv2.waitKey(25) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    break  
  print(f'Frame Rate  - {1/(time.time() - last_time)} fps.') 
  last_time = time.time()





































