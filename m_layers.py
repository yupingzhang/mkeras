from keras import backend as K
from keras.engine.topology import Layer
import numpy as np 
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints


def to_list(x):
    '''This normalizes a list/tensor into a list.
    If a tensor is passed, we return
    a list of size 1 containing the tensor.
    '''
    if type(x) is list:
        return x
    return [x]


def custom(x):
    #e = K.exp(x)
    #s = K.sum(e, axis=-1, keepdims=True)
    #return e / s
    return x


class Smooth(Layer):

    def __init__(self, 
                 activation=None,
                 use_bias=True,
                 kernel_initializer='uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.is_placeholder = False

        super(Smooth, self).__init__(**kwargs)

    def build(self, input_shape):
    	if input_shape is None:
        	raise RuntimeError('specify input shape')
        
        # print "building smooth layer: input shape:"
        # print input_shape
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(9, 9),
                                      initializer='uniform',
                                      trainable=True)

        super(Smooth, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input):
        output = K.dot(input, self.kernel)
        
        if self.activation is not None:
            return self.activation(output)
        else:
            return output


    def compute_output_shape(self, input_shape):
        return input_shape     # not changing the dimensions

