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

# a pool layer to pool values based on a dense weight matrix
class Densepool(Layer):

    def __init__(self, 
                 mtx,
                 output_dim,
                 activation=None,
                 **kwargs):
        self.mtx = mtx
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.trainable = False
        super(Densepool, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape is None:
            raise RuntimeError('specify input shape')

        super(Densepool, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input):
        # print "Densepool input >>> "
        dim = K.int_shape(input)
        print "Densepool mtx   >>> " 
        print self.mtx.shape
        weight = K.variable(value=self.mtx, name='pooling_kernal')
        print "Densepool weight variable >>> "
        print K.int_shape(weight)

        # output = K.transpose(input) * K.variable(value=self.mtx, name='pooling_kernal')   # multiply (element-wise) 
        #TODO
        tri = K.reshape(input, (dim[1], 3, 3))
        mtx_tensor = K.constant(self.mtx, dtype='float32', name='mtx_tensor')

        output = K.sum(mtx_tensor, axis=0)

        
        if self.activation is not None:
            return self.activation(output)
        else:
            return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, 3)


# 1.0/count
def compressmtx(face_mtx):
    c = [1.0/float(x) for x in K.sum(face_mtx, axis=1)]

    print "face_mtx shape >>> "
    print face_mtx.shape
    print "c shape >>> "
    print c.shape

    return c


