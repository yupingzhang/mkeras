from keras import backend as K
from keras.engine.topology import Layer
import numpy as np 
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints


class Smooth(Layer):

    def __init__(self, 
                 units,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='random_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.units = units
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
        
        # Create trainable weight variable for this layer.
        self.W = self.add_weight(name='weights',
                                      shape=(self.units, self.units),
                                      # initializer='random_normal',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(Smooth, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input):
        output = K.dot(input, self.W)    

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        
        if self.activation is not None:
            return self.activation(output)
        else:
            return output

    def compute_output_shape(self, input_shape):
        return input_shape     # not changing the dimensions

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'is_placeholder': False
        }

        base_config = super(Smooth, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# inputs: units: vel, cot, len, area...
# each input: tri * 3
# output: tri * 3
# wights: 1d-array of number of units
class Scaler(Layer):
    def __init__(self, 
                 units, 
                 **kwargs):
        self.units = units

        super(Scaler, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape is None:
            raise RuntimeError('specify input shape')
        
        # Create trainable weight variable for this layer.
        self.s = self.add_weight(name='scaler',
                                      shape=self.units, 
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

        super(Scaler, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs): # units * 
        output = inputs[0] * self.s[0]
        for i in range(1, len(inputs)):  
            output += inputs[i] * self.s[i]

        return output


# a pool layer to pool values based on a dense weight matrix
class Densepool(Layer):

    def __init__(self, 
                 mtx,
                 mtx_1,
                 activation=None,
                 **kwargs):
        self.mtx = mtx
        self.mtx_1 = mtx_1
        self.activation = activations.get(activation)
        self.trainable = False
        super(Densepool, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape is None:
            raise RuntimeError('specify input shape')

        super(Densepool, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input):
        dim = K.int_shape(input)
        flat_input = K.reshape(input, (-1, dim[1] * 6, 3))
        print "flat shape: >>>", flat_input.shape
        
        mtx_tensor = K.constant(self.mtx, dtype='float32', name='mtx_tensor')   # v_num x (tri*6)
        mtx_1_tensor = K.transpose(K.constant(self.mtx_1, dtype='float32', name='mtx_1_tensor'))                

        pos = K.dot(mtx_tensor, flat_input)   
        new_pos = K.permute_dimensions(pos, (1, 0, 2))
        # print new_pos.shape   # (?, 700, 3)
        
        tri = K.dot(mtx_1_tensor, new_pos) 
        output = K.permute_dimensions(tri, (1, 0, 2)) 
        s = K.int_shape(output)
        # output = K.reshape(output, (-1, s[1]/3, 9))
        output = K.reshape(output, (-1, dim[1], dim[2]))

        if self.activation is not None:     
            return self.activation(output)
        else:
            return output

    def compute_output_shape(self, input_shape):
        return input_shape



