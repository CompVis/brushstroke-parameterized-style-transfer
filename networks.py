import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
import pickle


class VGG:

    def __init__(self, ckpt_path):
        self.param_dict = pickle.load(open(ckpt_path, 'rb'))

    def extract_features(self, x):
        features = {}
        x = self._conv2d_block(x, self.param_dict['block1']['conv1']['weight'], self.param_dict['block1']['conv1']['bias'])
        features['conv1_1'] = x
        x = self._conv2d_block(x, self.param_dict['block1']['conv2']['weight'], self.param_dict['block1']['conv2']['bias'])
        features['conv1_2'] = x
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
        features['conv1_2_pool'] = x

        x = self._conv2d_block(x, self.param_dict['block2']['conv1']['weight'], self.param_dict['block2']['conv1']['bias'])
        features['conv2_1'] = x
        x = self._conv2d_block(x, self.param_dict['block2']['conv2']['weight'], self.param_dict['block2']['conv2']['bias'])
        features['conv2_2'] = x
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
        features['conv2_2_pool'] = x

        x = self._conv2d_block(x, self.param_dict['block3']['conv1']['weight'], self.param_dict['block3']['conv1']['bias'])
        features['conv3_1'] = x
        x = self._conv2d_block(x, self.param_dict['block3']['conv2']['weight'], self.param_dict['block3']['conv2']['bias'])
        features['conv3_2'] = x
        x = self._conv2d_block(x, self.param_dict['block3']['conv3']['weight'], self.param_dict['block3']['conv3']['bias'])
        features['conv3_3'] = x
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
        features['conv3_3_pool'] = x

        x = self._conv2d_block(x, self.param_dict['block4']['conv1']['weight'], self.param_dict['block4']['conv1']['bias'])
        features['conv4_1'] = x
        x = self._conv2d_block(x, self.param_dict['block4']['conv2']['weight'], self.param_dict['block4']['conv2']['bias'])
        features['conv4_2'] = x
        x = self._conv2d_block(x, self.param_dict['block4']['conv3']['weight'], self.param_dict['block4']['conv3']['bias'])
        features['conv4_3'] = x
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
        features['conv4_3_pool'] = x

        x = self._conv2d_block(x, self.param_dict['block5']['conv1']['weight'], self.param_dict['block5']['conv1']['bias'])
        features['conv5_1'] = x
        x = self._conv2d_block(x, self.param_dict['block5']['conv2']['weight'], self.param_dict['block5']['conv2']['bias'])
        features['conv5_2'] = x
        x = self._conv2d_block(x, self.param_dict['block5']['conv3']['weight'], self.param_dict['block5']['conv3']['bias'])
        features['conv5_3'] = x
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
        features['conv5_3_pool'] = x
        return features

    def _conv2d_block(self, x, kernel, bias):
        x = tf.nn.conv2d(x, filters=kernel, strides=[1, 1], padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
        x = tf.nn.bias_add(x, bias=bias, data_format='N...C')
        x = tf.nn.relu(x)
        return x

