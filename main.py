from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features, [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  last = pool1
  for i in range(0, 10):
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=last,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    last = pool2

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)

prediction = None

def main(unused_argv):
  # Load training and eval data
  
  # training_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename="trainX.csv",target_dtype=np.float32,features_dtype=np.float32)
  # train_data = np.append(training_set.data, np.transpose(np.array([training_set.target])), axis=1)
  # # print("hello")
  # training_labels = tf.contrib.learn.datasets.base.load_csv_without_header(filename="trainY.csv",target_dtype=np.float32,features_dtype=np.float32)
  # train_labels = training_labels.data
  # train_labels = [np.argmax(arr) for arr in train_labels]

  mnist_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="mnist_convnet_model5")
  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
#   logging_hook = tf.train.LoggingTensorHook(
#       tensors=tensors_to_log, every_n_iter=1000)
#   mnist_classifier.fit(
#     x=train_data,
#     y=train_labels,
#     batch_size=100,
#     steps=20000,
#     monitors=[logging_hook])
#   metrics = {
#     "accuracy":
#         learn.MetricSpec(
#             metric_fn=tf.metrics.accuracy, prediction_key="classes"),
# }
#   eval_results = mnist_classifier.evaluate(
#     x=train_data, y=train_labels, metrics=metrics)
#   print(eval_results)

  testing_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename="testX.csv",target_dtype=np.float32,features_dtype=np.float32)
  test_data = np.append(testing_set.data, np.transpose(np.array([testing_set.target])), axis=1)

  # prediction = list(mnist_classifier.predict(x=np.array([[np.float32(0.5) for _ in range(0, 784)], [np.float32(0.6) for _ in range(0, 784)]])))
  
  prediction = list(mnist_classifier.predict(x=test_data))
  prediction = [x['probabilities'] for x in prediction]

  ret = [[int(i), int(np.argmax(arr))] for (i, arr) in zip(range(0,len(prediction)),prediction)]
  np.savetxt("testY.csv", ret, delimiter=",", fmt='%i', header="id,digit")

main("hello")