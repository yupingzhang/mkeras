from keras import backend as K
from keras.engine.topology import Layer
import numpy as np 
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints


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

# ================================================================
# mtx: np.array
def pooling_kernal_init(shape, mtx):
    return K.variable(value=mtx, dtype='float64', name='pooling_kernal')


# a pool layer to pool values based on a dense weight matrix
class Densepool(Layer):

    def __init__(self, 
                 activation=None,
                 kernel_initializer=pooling_kernal_init,
                 **kwargs):
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.trainable = False
      super(Densepool, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape is None:
            raise RuntimeError('specify input shape')

        # Create a non-trainable weight variable  
        self.kernel = self.add_weight(name='poolingkernel',
                                      shape=(input_shape[0],),
                                      initializer=kernel_initializer,
                                      trainable=False)

        super(Densepool, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input):
        output = input * self.kernel   # multiply (element-wise) 
        if self.activation is not None:
            return self.activation(output)
        else:
            return output

    def compute_output_shape(self, input_shape):
        return input_shape 

