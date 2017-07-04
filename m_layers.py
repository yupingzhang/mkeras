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
                 # use_bias=True,
                 # kernel_initializer='glorot_uniform',
                 # bias_initializer='zeros',
                 # kernel_regularizer=None,
                 # bias_regularizer=None,
                 # activity_regularizer=None,
                 # kernel_constraint=None,
                 # bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        
        self.activation = activations.get(activation)
        # self.use_bias = use_bias
        # self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        # self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # self.bias_regularizer = regularizers.get(bias_regularizer)
        # self.activity_regularizer = regularizers.get(activity_regularizer)
        # self.kernel_constraint = constraints.get(kernel_constraint)
        # self.bias_constraint = constraints.get(bias_constraint)
        self.is_placeholder = False
        super(Smooth, self).__init__(**kwargs)


    def build(self, input_shape):
    	if input_shape is None:
        	raise RuntimeError('specify input shape')
        
        print "building smooth layer: input shape:"
        print input_shape
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(9, 9),
                                      initializer='uniform',
                                      trainable=True)

        super(Smooth, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input):
        # inputs = to_list(input)
		# parse each triangle
        # for i in range(1, len(input)):
        #     ttri = input[i]
        #     print ttri
        #     v1 = ttri[0:3]
        #     v2 = ttri[3:6]
        #     v3 = ttri[6:9]
        #     # add subdivided triangles to the list
        #     vert_1 = K.dot(self.kernel, K.transpose(v1.extend(v2).extend(v3)))
        #     vert_2 = K.dot(self.kernel, K.transpose(v2.extend(v1).extend(v3)))
        #     vert_3 = K.dot(self.kernel, K.transpose(v3.extend(v1).extend(v2)))
        #     # add updated triangle
        #     tri = vert_1.extend(vert_2).extend(vert_3)
        #     tri_list.append(tri)
        
        # print "call smooth layer..."
        # print K.shape(input)
        # print ">>>\n"
        # print inputs[0]
        # for x_elem in inputs:
        #     print K.int_shape(x_elem)

        output = K.dot(input, self.kernel)
        
        if self.activation is not None:
            return self.activation(output)
        else:
            return output


    def compute_output_shape(self, input_shape):
        # return (input_shape[0], input_shape[1:])   # not changing the dimensions
        return input_shape

