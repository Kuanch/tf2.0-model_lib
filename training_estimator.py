from __future__ import absolute_import, print_function, division
from datetime import datetime
import os

import tensorflow as tf

from model_factory import get_model_fn
from dataset import create_dataset
from preprocess import preprocess



def create_estimator(params, run_config):
    
  model = get_model_fn(params)
  print(model.summary())
  
  optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  mnist_classifier = tf.keras.estimator.model_to_estimator(
      keras_model=keras_model,
      config=run_config
  )
  
  return classifier



def run_experiment(hparams, train_data, train_labels, run_config, create_estimator_fn=create_estimator):
  train_spec = tf.estimator.TrainSpec(
      input_fn = tf.estimator.inputs.numpy_input_fn(
          x={'input_image': train_data},
          y=train_labels,
          batch_size=hparams.batch_size,
          num_epochs=None,
          shuffle=True),
      max_steps=hparams.max_training_steps
  )

  eval_spec = tf.estimator.EvalSpec(
      input_fn = tf.estimator.inputs.numpy_input_fn(
          x={'input_image': train_data},
          y=train_labels,
          batch_size=hparams.batch_size,
          num_epochs=1,
          shuffle=False),
      steps=None,
      throttle_secs=hparams.eval_throttle_secs
  )

  tf.logging.set_verbosity(tf.logging.INFO)

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
