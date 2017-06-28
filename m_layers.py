from keras import backend as K
from keras.engine.topology import Layer

class Smooth(Layer):

    def __init__(self, **kwargs):
        super(Smooth, self).__init__(**kwargs)

    def build(self, input_shape):
    	if input_shape is None:
        	raise RuntimeError('specify input shape')

        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(9, 9),
                                      initializer='uniform',
                                      trainable=True)
        super(Smooth, self).build(input_shape)  # Be sure to call this somewhere!



    def call(self, input, mask=None):
        tri_list = np.array()
		# parse each triangle
        for i in range(1, len(input)):
            ttri = input[i]
            print ttri
            v1 = input[0:3]
            v2 = input[3:6]
            v3 = input[6:9]

            # add subdivided triangles to the list
            vert_1 = K.dot(self.kernel, K.transpose(v1.extend(v2).extend(v3)))
            vert_2 = K.dot(self.kernel, K.transpose(v2.extend(v1).extend(v3)))
            vert_3 = K.dot(self.kernel, K.transpose(v3.extend(v1).extend(v2)))
            # add updated triangle
            tri = vert_1.extend(vert_2).extend(vert_3)
            tri_list.append(tri)

        return tri_list

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[0])   # not changing the dimensions

