from __future__ import absolute_import, division, print_function
import tensorflow as tf



def create_dataset(tfrecord_path, batch_size=32, num_epoch=1, train_image_size=224, num_label,
                   preprocess_fn, shuffle_buffer=100, is_training=True, 
                   cifar10_mode=False):

 
  if cifar10_mode:
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    cifar10_mode = True
    tf.print('Cifar10 mode')


    dataset = dataset.map(lambda image, label, image_name:preprocess_fn(image, label,
                                       image_name, train_image_size, train_image_size,
                                       num_label, is_training=is_training,
                                       cifar10_mode=cifar10_mode),
                                       num_parallel_calls=8)


  # Set the number of datapoints you want to load and shuffle 
  dataset = dataset.shuffle(shuffle_buffer)

  # This dataset will go on forever
  dataset = dataset.repeat(num_epoch)
  
  dataset = dataset.prefetch(buffer_size=batch_size*2)
  
  # Set the batchsize
  dataset = dataset.batch(batch_size, drop_remainder=False)
    
  return dataset