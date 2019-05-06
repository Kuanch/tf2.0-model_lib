"""Inception v4.

Architecture as described in https://arxiv.org/pdf/1602.07261.pdf

This is a Tensorflow 2.0 prototype, might still having some optimization and coding styly issues.

Convlution layer default parameters (filters=None, kernel_size=None, stride=(1, 1), padding='valid')
Average Pooling default parameters (pool_size=(2, 2), stride=None)
Max Pooling default parameters (pool_size=(2, 2), stride=None)

Keras Model Usage:
  model = InceptionV4(num_classes=1000)
  model.compile(optimizer, loss, metrics)
  
  # Training
  history = model.fit(x_train, y_train, batch_size, epoch, validation_data=(x_val, yval))

  # Print out loss and metrics during training
  print(history.history)

  # Evaluation on test set
  results = models.evaluate(x_test, y_test, batch_size)

  # Prediction
  predictions = model.predict(x_inference)

  # Save model
  keras.experimental.export_saved_model(model, 'path_to_saved_model')

  # Recreate model
  new_model = keras.experimental.load_from_saved_model('path_to_saved_model')

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

from tensorflow.keras.layers import Layer, Conv2D, AveragePooling2D, MaxPool2D, \
                                    BatchNormalization, ReLU, Dropout, Flatten, \
                                    Dense, Softmax



__all__ = ['Stem', 'InceptionModule_A', 'InceptionModule_B', 'InceptionModule_C', 'ReductionModule_A', \
          'ReductionModule_B', 'InceptionV4']




class Stem(tf.keras.layers.Layer):
    """Stem part of InceptionV4"""
    def __init__(self, input_channels):
      super(Stem, self).__init__()
      self.input_channels = input_channels
      
      self.straight_3x3_0 = Conv2D(filters=self.input_channels, kernel_size=(3, 3), strides=(2, 2))
      self.straight_3x3_1 = Conv2D(filters=self.input_channels, kernel_size=(3, 3))
      self.straight_3x3_2 = Conv2D(filters=self.input_channels*2, kernel_size=(3, 3), padding='same')

      self.branch_0_0 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))
      self.branch_1_0 = Conv2D(filters=self.input_channels*3, kernel_size=(3, 3), strides=(2, 2))
      # Concate branch 0 and 1

      self.branch_0_1 = Conv2D(filters=self.input_channels*2, kernel_size=(1, 1), padding='same')
      self.branch_0_2 = Conv2D(filters=self.input_channels*3, kernel_size=(3, 3))

      self.branch_1_1 = Conv2D(filters=self.input_channels*2, kernel_size=(1, 1), padding='same')
      self.branch_1_2 = Conv2D(filters=self.input_channels*2, kernel_size=(7, 1), padding='same')
      self.branch_1_3 = Conv2D(filters=self.input_channels*2, kernel_size=(1, 7), padding='same')
      self.branch_1_4 = Conv2D(filters=self.input_channels*3, kernel_size=(3, 3))
      # Concate branch 0 and 1

      self.branch_0_3 = Conv2D(filters=self.input_channels*6, kernel_size=(3, 3), strides=(2, 2))
      self.branch_1_5 = MaxPool2D(strides=(2, 2))

    
    # Would be slower after tf.function decorated on CPU.
    @tf.function
    def call(self, inputs):
      return self.stem_block(inputs)


    def stem_block(self, x):
      x = self.straight_3x3_0(x)
      x = self.straight_3x3_1(x)
      x = self.straight_3x3_2(x)
      branch_0 = self.branch_0_0(x)
      branch_1 = self.branch_1_0(x)

      concat = tf.concat([branch_0, branch_1], axis=3)

      branch_0 = self.branch_0_1(concat)
      branch_0 = self.branch_0_2(branch_0)

      branch_1 = self.branch_1_1(concat)
      branch_1 = self.branch_1_2(branch_1)
      branch_1 = self.branch_1_3(branch_1)
      branch_1 = self.branch_1_4(branch_1)

      concat = tf.concat([branch_0, branch_1], axis=3)

      branch_0 = self.branch_0_3(concat)
      branch_1 = self.branch_1_5(concat)

      return tf.concat([branch_0, branch_1], axis=3)



class InceptionModule_A(tf.keras.layers.Layer):
  """Module A of InceptionV4"""
  def __init__(self, input_channels):
    super(InceptionModule_A, self).__init__()
    self.input_channels = input_channels

    # Branch 0
    self.avg_pool_b0 = AveragePooling2D(pool_size=(1, 1), padding='same')
    self.conv_1x1_b0 = Conv2D(filters=self.input_channels*3, kernel_size=(1, 1), padding='same')
    
    # Branch 1
    self.conv_1x1_b1 = Conv2D(filters=self.input_channels*3, kernel_size=(1, 1), padding='same')

    # Branch 2
    self.conv_1x1_b2 = Conv2D(filters=self.input_channels*2, kernel_size=(1, 1), padding='same')
    self.conv_3x3_b2 = Conv2D(filters=self.input_channels*3, kernel_size=(3, 3), padding='same')
    
    # Branch 3
    self.conv_1x1_b3 = Conv2D(filters=self.input_channels*2, kernel_size=(1, 1), padding='same')
    self.conv_3x3_b3 = Conv2D(filters=self.input_channels*3, kernel_size=(3, 3), padding='same')
    self.conv_3x3_b3_2 = Conv2D(filters=self.input_channels*3, kernel_size=(3, 3), padding='same')


  # Would be slower after tf.function decorated on CPU.
  @tf.function
  def call(self, inputs):
    return self.wide_block(inputs)


  def wide_block(self, x):
    b0 = self.avg_pool_b0(x)
    b0 = self.conv_1x1_b0(b0)

    b1 = self.conv_1x1_b1(x)

    b2 = self.conv_1x1_b2(x)
    b2 = self.conv_3x3_b2(b2)

    b3 = self.conv_1x1_b3(x)
    b3 = self.conv_3x3_b3(b3)
    b3 = self.conv_3x3_b3_2(b3)

    return tf.concat([b0, b1, b2, b3], axis=3)



class InceptionModule_B(tf.keras.layers.Layer):
  """Module B fo InceptionV4"""
  def __init__(self, input_channels):
    super(InceptionModule_B, self).__init__()
    self.input_channels = input_channels

    # Branch 0
    self.avg_pool_b0 = AveragePooling2D(pool_size=(1, 1), padding='same')
    self.conv_1x1_b0 = Conv2D(filters=self.input_channels*4, kernel_size=(1, 1), padding='same')

    # Branch 1
    self.conv_1x1_b1 = Conv2D(filters=self.input_channels*12, kernel_size=(1, 1), padding='same')

    # Branch 2
    self.conv_1x1_b2 = Conv2D(filters=self.input_channels*6, kernel_size=(1, 1), padding='same')
    self.conv_1x7_b2 = Conv2D(filters=self.input_channels*7, kernel_size=(1, 7), padding='same')
    self.conv_7x1_b2 = Conv2D(filters=self.input_channels*8, kernel_size=(7, 1), padding='same')

    # Branch 3
    self.conv_1x1_b3 = Conv2D(filters=self.input_channels*6, kernel_size=(1, 1), padding='same')
    self.conv_1x7_b3 = Conv2D(filters=self.input_channels*6, kernel_size=(1, 7), padding='same')
    self.conv_7x1_b3 = Conv2D(filters=self.input_channels*7, kernel_size=(7, 1), padding='same')
    self.conv_1x7_b3_2 = Conv2D(filters=self.input_channels*7, kernel_size=(1, 7), padding='same')
    self.conv_7x1_b3_2 = Conv2D(filters=self.input_channels*8, kernel_size=(7, 1), padding='same')


  # Would be slower after tf.function decorated on CPU.
  @tf.function
  def call(self, inputs):
    return self.wide_block(inputs)


  def wide_block(self, x):
    b0 = self.avg_pool_b0(x)
    b0 = self.conv_1x1_b0(b0)

    b1 = self.conv_1x1_b1(x)

    b2 = self.conv_1x1_b2(x)
    b2 = self.conv_1x7_b2(b2)
    b2 = self.conv_7x1_b2(b2)

    b3 = self.conv_1x1_b3(x)
    b3 = self.conv_1x7_b3(b3)
    b3 = self.conv_7x1_b3(b3)
    b3 = self.conv_1x7_b3_2(b3)
    b3 = self.conv_7x1_b3_2(b3)

    return tf.concat([b0, b1, b2, b3], axis=3)



class InceptionModule_C(tf.keras.layers.Layer):
  """Module_C of InceptionV4"""
  def __init__(self, input_channels):
    super(InceptionModule_C, self).__init__()
    self.input_channels = input_channels
    # Branch 0
    self.avg_pool_b0 = AveragePooling2D(pool_size=(1, 1), padding='same')
    self.conv_1x1_b0 = Conv2D(filters=self.input_channels*4, kernel_size=(1, 1), padding='same')

    # Branch 1
    self.conv_1x1_b1 = Conv2D(filters=self.input_channels*4, kernel_size=(1, 1), padding='same')

    # Branch 2
    self.conv_1x1_b2 = Conv2D(filters=self.input_channels*6, kernel_size=(1, 1), padding='same')
    self.conv_1x3_b2 = Conv2D(filters=self.input_channels*4, kernel_size=(1, 3), padding='same')
    self.conv_3x1_b2 = Conv2D(filters=self.input_channels*4, kernel_size=(3, 1), padding='same')

    # Branch 3
    self.conv_1x1_b3 = Conv2D(filters=self.input_channels*6, kernel_size=(1, 1), padding='same')
    self.conv_1x3_b3 = Conv2D(filters=self.input_channels*7, kernel_size=(1, 3), padding='same')
    self.conv_3x1_b3 = Conv2D(filters=self.input_channels*8, kernel_size=(3, 1), padding='same')
    self.conv_1x3_b3_2 = Conv2D(filters=self.input_channels*4, kernel_size=(1, 3), padding='same')
    self.conv_3x1_b3_2 = Conv2D(filters=self.input_channels*4, kernel_size=(3, 1), padding='same')


  # Would be slower after tf.function decorated on CPU.
  @tf.function
  def call(self, inputs):
    return self.wide_block(inputs)


  def wide_block(self, x):
    b0 = self.avg_pool_b0(x)
    b0 = self.conv_1x1_b0(b0)

    b1 = self.conv_1x1_b1(x)

    b2 = self.conv_1x1_b2(x)
    b2_0 = self.conv_1x3_b2(b2)
    b2_1 = self.conv_3x1_b2(b2)

    b3 = self.conv_1x1_b3(x)
    b3 = self.conv_1x3_b3(b3)
    b3 = self.conv_3x1_b3(b3)
    b3_0 = self.conv_3x1_b3_2(b3)
    b3_1 = self.conv_1x3_b3_2(b3)

    return tf.concat([b0, b1, b2_0, b2_1, b3_0, b3_1], axis=3)



class ReductionModule_A(tf.keras.layers.Layer):
  """Reduction A of InceptionV4"""
  def __init__(self, input_channels):
    super(ReductionModule_A, self).__init__()
    self.input_channels = input_channels
    # Branch 0
    self.max_pool_b0 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))

    # Branch 1
    self.conv_3x3_b1 = Conv2D(filters=self.input_channels*6, kernel_size=(3, 3), strides=(2, 2))

    # Branch 2
    self.conv_1x1_b2 = Conv2D(filters=self.input_channels*7, kernel_size=(1, 1), padding='same')
    self.conv_3x3_b2 = Conv2D(filters=self.input_channels*8, kernel_size=(3, 3), padding='same')
    self.conv_3x3_b2_2 = Conv2D(filters=self.input_channels*12, kernel_size=(3, 3), strides=(2, 2))


  # Would be slower after tf.function decorated on CPU.
  @tf.function
  def call(self, x):
    return self.reduction_block(x)


  def reduction_block(self, x):
    b0 = self.max_pool_b0(x)

    b1 = self.conv_3x3_b1(x)

    b2 = self.conv_1x1_b2(x)
    b2 = self.conv_3x3_b2(b2)
    b2 = self.conv_3x3_b2_2(b2)

    return tf.concat([b0, b1, b2], axis=3)



class ReductionModule_B(tf.keras.layers.Layer):
  """Reduction B of InceptionV4"""
  def __init__(self, input_channels):
    super(ReductionModule_B, self).__init__()
    self.input_channels = input_channels
    # Branch 0
    self.max_pool_b0 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))

    # Branch 1
    self.conv_1x1_b1 = Conv2D(filters=self.input_channels*3, kernel_size=(1, 1), padding='same')
    self.conv_3x3_b1 = Conv2D(filters=self.input_channels*3, kernel_size=(3, 3), strides=(2, 2))

    # Branch 2
    self.conv_1x1_b2 = Conv2D(filters=self.input_channels*4, kernel_size=(1, 1), padding='same')
    self.conv_1x7_b2 = Conv2D(filters=self.input_channels*4, kernel_size=(1, 7), padding='same')
    self.conv_7x1_b2 = Conv2D(filters=self.input_channels*5, kernel_size=(7, 1), padding='same')
    self.conv_3x3_b2 = Conv2D(filters=self.input_channels*4, kernel_size=(3, 3), strides=(2, 2))


  # Would be slower after tf.function decorated on CPU.
  @tf.function
  def call(self, inputs):
    return self.reduction_block(inputs)


  def reduction_block(self, x):
    b0 = self.max_pool_b0(x)

    b1 = self.conv_1x1_b1(x)
    b1 = self.conv_3x3_b1(b1)

    b2 = self.conv_1x1_b2(x)
    b2 = self.conv_1x7_b2(b2)
    b2 = self.conv_7x1_b2(b2)
    b2 = self.conv_3x3_b2(b2)

    return tf.concat([b0, b1, b2], axis=3)
    
    


class InceptionV4(tf.keras.Model):
  """Body of InceptionV4"""
  def __init__(self, num_classes=None, num_filter_base=32):
    super(InceptionV4, self).__init__()
    self.num_filter_base = num_filter_base
    self.num_classes = num_classes

    # Extractor part
    self.body = [Stem(self.num_filter_base),
    # 299x299x3
    InceptionModule_A(self.num_filter_base),
    InceptionModule_A(self.num_filter_base),
    InceptionModule_A(self.num_filter_base),
    InceptionModule_A(self.num_filter_base),
    # 35x35x384
    ReductionModule_A(self.num_filter_base),
    # 17x17x1024
    InceptionModule_B(self.num_filter_base),
    InceptionModule_B(self.num_filter_base),
    InceptionModule_B(self.num_filter_base),
    InceptionModule_B(self.num_filter_base),
    InceptionModule_B(self.num_filter_base),
    InceptionModule_B(self.num_filter_base),
    InceptionModule_B(self.num_filter_base),
    # 17x17x1024
    ReductionModule_B(self.num_filter_base*2),
    # 8x8x1536
    InceptionModule_C(self.num_filter_base*2),
    InceptionModule_C(self.num_filter_base*2),
    InceptionModule_C(self.num_filter_base*2),
    # 8x8x1536
    ]

    # Add logit
    if self.num_classes is not None:
      self.body += [AveragePooling2D(pool_size=(8, 8), data_format='channels_last'), # 1x1x1536
                    Dropout(rate=0.8), Flatten(), Dense(units=self.num_classes), Softmax()]
    # Output shape: (batch_size, num_classes)


  # Would be slower after tf.function decorated on CPU.
  @tf.function
  def call(self, inputs):
    return self.build(num_classes=self.num_classes)(inputs)


  def build(self, num_classes, input_shape=[None, None, 3]):
    return tf.keras.Sequential(self.body)



# Graph flow test
if __name__ == '__main__':
  x = tf.ones([10, 299, 299, 3], dtype=tf.float32)
  output = InceptionV4(num_classes=1000)(x)
  print(output.get_shape())