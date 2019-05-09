from __future__ import absolute_import, print_function, division
from datetime import datetime
import os

import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from model.MobileNetV2 import *

from collections import namedtuple

from model.model_factory import get_model_fn
from dataset.create_dataset import create_dataset
from preprocess.preprocess import preprocess_for_train


TrainParameters = namedtuple('TrainParameters', ['model_name', 'num_classes', 'learning_rate',
                                                 'max_training_steps', 'max_eval_steps','eval_throttle_secs'])



model = tf.keras.Sequential([Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same'),
           # 112x112x32
           Bottleneck(filters=16, strides=(1, 1), expansion=1),
           # 112x112x16
           Bottleneck(filters=24, strides=(2, 2), expansion=6),
           Bottleneck(filters=24, strides=(1, 1), expansion=6),
           # 56x56x24
           Bottleneck(filters=32, strides=(2, 2), expansion=6),
           Bottleneck(filters=32, strides=(1, 1), expansion=6),
           Bottleneck(filters=32, strides=(1, 1), expansion=6),
           # 28x28x32
           Bottleneck(filters=64, strides=(2, 2), expansion=6),
           Bottleneck(filters=64, strides=(1, 1), expansion=6),
           Bottleneck(filters=64, strides=(1, 1), expansion=6),
           Bottleneck(filters=64, strides=(1, 1), expansion=6),
           # 14x14x64
           Bottleneck(filters=96, strides=(1, 1), expansion=6),
           Bottleneck(filters=96, strides=(1, 1), expansion=6),
           Bottleneck(filters=96, strides=(1, 1), expansion=6),
           # 14x14x96
           Bottleneck(filters=160, strides=(2, 2), expansion=6),
           Bottleneck(filters=160, strides=(1, 1), expansion=6),
           Bottleneck(filters=160, strides=(1, 1), expansion=6),
           # 7x7x160
           Bottleneck(filters=320, strides=(1, 1), expansion=6),
           # 7x7x320
           Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1))
           # 7x7x1280
           ])



def create_estimator(params, run_config):
    
  #model = get_model_fn(params.model_name, params.num_classes)
  
  optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  classifier = tf.keras.estimator.model_to_estimator(
      keras_model=model,
      config=run_config
  )
  
  return classifier



def run_experiment(hparams, train_dataset, eval_dataset, run_config, create_estimator_fn=create_estimator):
  train_spec = tf.estimator.TrainSpec(
      input_fn = train_dataset,
      max_steps=hparams.max_training_steps
  )

  eval_spec = tf.estimator.EvalSpec(
      input_fn = eval_dataset,
      steps=hparams.max_eval_steps,
      #throttle_secs=hparams.eval_throttle_secs
  )

  time_start = datetime.utcnow()
  print('Experiment started at {} \n'.format(time_start.strftime('%H:%M:%S')))

  estimator = create_estimator_fn(hparams, run_config)

  tf.estimator.train_and_evaluate(
      estimator=estimator,
      train_spec=train_spec,
      eval_spec=eval_spec
  )

  time_end = datetime.utcnow()
  print('Experiment finished at {} \n'.format(time_end.strftime('%H:%M:%S')))
  time_elapsed = time_end - time_start
  print('Experiment elapsed time: {} seconds \n'.format(time_elapsed.total_seconds()))

  return estimator



if __name__ == '__main__':
  default_params = TrainParameters(model_name='mobilenet_v2', num_classes=10, learning_rate=0.045,
                                   max_training_steps=100, max_eval_steps=100, eval_throttle_secs=20)

  run_config = tf.estimator.RunConfig(
    tf_random_seed=19830610,
  )


  train_dataset = lambda: create_dataset(tfrecord_path=None, num_label=default_params.num_classes, preprocess_fn=preprocess_for_train, cifar10_mode=True)
  eval_dataset = lambda: create_dataset(tfrecord_path=None, num_label=default_params.num_classes, preprocess_fn=preprocess_for_train, cifar10_mode=True, is_training=False)


  run_experiment(hparams=default_params, train_dataset=train_dataset, eval_dataset=eval_dataset, run_config=run_config)

