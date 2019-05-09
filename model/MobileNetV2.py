from __future__ import print_function, division, absolute_import

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ReLU, BatchNormalization, AveragePooling2D, Dropout, Softmax

from collections import namedtuple


__all__ = ['Bottleneck', 'MobilenetV2']


BottleneckBlock = namedtuple('Bottleneck', ['filters', 'strides', 'expansion'])



class Squeeze(tf.keras.layers.Layer):
	def __init__(self):
		super(Squeeze, self).__init__()


	def call(self, inputs):
		return tf.squeeze(inputs, axis=[1, 2])



class Bottleneck(tf.keras.layers.Layer):
	"""docstring for Bottleneck"""
	def __init__(self, filters, strides=(1, 1), expansion=1):
		super(Bottleneck, self).__init__()
		self.filters = filters
		self.strides = strides
		self.expansion = expansion

		

	def bottleneck(self, inputs):
		x = self.pointwise_conv(inputs)
		x = self.depthwise_conv(x)
		output = self.pointwise_conv_2(x)

		if (self.strides == (1, 1) and \
						inputs.get_shape().as_list()[3] == output.get_shape().as_list()[3]):

			print('resudial activate')
			output += inputs

		return output



	def build(self, input_shape=[None, None, 3]):
		self.input_channels = input_shape.as_list()[-1]
		self.expand_channel = self.expansion * self.input_channels

		self.pointwise_conv = Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(1, 1),
																 activation = ReLU(max_value=6), kernel_regularizer=tf.keras.regularizers.l2(0.00004))
		self.batch_norm_0 = BatchNormalization(axis=3, momentum=0.9997, scale=False)

		
		self.depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=self.strides, padding='same',
																					activation = ReLU(max_value=6), depthwise_regularizer=tf.keras.regularizers.l2(0.00004))
		self.batch_norm_depthwise = BatchNormalization(axis=3, momentum=0.9997, scale=False)


		self.pointwise_conv_2 = Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(1, 1),
																 activation = ReLU(max_value=6), kernel_regularizer=tf.keras.regularizers.l2(0.00004))
		self.batch_norm_1 = BatchNormalization(axis=3, momentum=0.9997, scale=False)



	@tf.function
	def call(self, inputs):
		return self.bottleneck(inputs)



class MobilenetV2(tf.keras.Sequential):
	"""docstring for MobilenetV2"""
	def __init__(self, num_classes=None):
		super(MobilenetV2, self).__init__()
		self.num_classes = num_classes
					 # 224x224x3
		self.body = [Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same'),
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
					 ]


		# Add logit
		if self.num_classes is not None:
			self.body += [AveragePooling2D(pool_size=(7, 7), data_format='channels_last'), # 1x1x1280
			Dropout(rate=0.8), Conv2D(filters=self.num_classes, kernel_size=(1, 1)), Squeeze(), Softmax()]
			# Output shape: (batch_size, num_classes)

		for layer in self.body:
			self.add(layer)


	def add(self, layer):
	  super(MobilenetV2, self).add(layer)
        

# Default preprocessing size from origin parper
MobilenetV2.default_image_size = 224



# Flow test
if __name__ == '__main__':
	x = tf.ones([10, 224, 224, 3], dtype=tf.float32)
	model = MobilenetV2(num_classes=10)
	output = model(x)
	print(x.get_shape().as_list(), output.get_shape().as_list())
