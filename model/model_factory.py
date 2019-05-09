from __future__ import absolute_import, print_function, division
import functools

import tensorflow as tf

from model import InceptionV4
from model import MobileNetV2


model_map = {'inception_v4':InceptionV4.InceptionV4,
              'mobilenet_v2':MobileNetV2.MobilenetV2}



def get_model_fn(model_name, num_classes):
  if model_name not in model_map:
    raise ValueError('Name of network unknown %s' % model_name)

  model_class = model_map[model_name](num_classes=num_classes)

  return model_class


