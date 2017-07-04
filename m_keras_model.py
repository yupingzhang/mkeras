import argparse
import os.path
import operator
import numpy as np 
from keras import utils
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Activation, Embedding
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from m_layers import Smooth, custom



# dim = []
# model = Sequential()

path_coarse = 'sim_coarse/053d/'
path_tracking = 'sim_tracking/053/'


# load data
def obj_parser(file_name, batch_data, dim):
    # position v, velocity nv
    dim = 0 
    if not os.path.isfile(file_name):
        print "file not exist"
        return
    
    vert = []
    sample = []
    with open(file_name, "r") as f1:
        for line in f1:
            s = line.strip().split(' ')
            if s[0] == 'v':
                vert.extend(map(float, s[1:]))
            elif s[0] == 'nv':
                vert.extend(map(float, s[1:]))
            if len(vert) == 6:
                dim = dim + 1
                sample.append(vert)
                vert = []

    batch_data.append(sample)


# parse data as 9*tri_num (position only)
def obj2tri(file_name, batch_data):
    if not os.path.isfile(file_name):
        print "file not exist"
        return
    vert = []
    data = []
    with open(file_name, "r") as f1:
        for line in f1:
            s = line.strip().split(' ')
            if s[0] == 'v':
                vert.append(map(float, s[1:]))
            #elif s[0] == 'nv':
            #    vert.extend(map(float, s[1:]))
            elif s[0] == 'f':
                face = []
                id1 = int(s[1].strip().split('/')[0]) - 1  # index start at 1
                id2 = int(s[2].strip().split('/')[0]) - 1
                id3 = int(s[3].strip().split('/')[0]) - 1
                face.extend(vert[id1])
                face.extend(vert[id2])
                face.extend(vert[id3])
                data.append(face)
    batch_data.append(data)


def setmodel():

    model = Sequential()
    model.add(Smooth(input_shape=(1292,9), name='smoo_layer'))
    
    # sim_input = Input(shape=(1292,9), dtype='float32', name='sim_input')
    #input_tensor = Input(shape=(1292,9), name='sim_input')
    # smooth_layer = Smooth(activation=custom)

    #smooth_layer = Smooth(shape=(1292,9), name='smoo_layer')
    #smooth_layer.trainable = True
    # smooth_out = Smooth(custom)(sim_input)
    #out_tensor = smooth_layer(input_tensor)

    # x = Dense(9, activation='relu', name="dense_one")(o_tensor)

    # model = Model(inputs=sim_input, outputs=smooth_out)
    #model = Model(input_tensor, out_tensor)

    # print model.get_layer(index=1).name # smooth_1

    # model.compile(loss='mean_squared_error', optimizer='sgd')
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse', 'acc'])

    print "set model and compile"
    model.summary()

    return model


def train(model):
    batch_coarse = []
    batch_fine = []
    bath_delta = []
    #dim_c = 0
    #dim_f = 0

    for x in xrange(1,101):
        file_name = str(x).zfill(5) + '_00.obj'
        coarse_file = path_coarse + file_name
        fine_file = path_tracking + file_name  # + path_index 
        #obj_parser(coarse_file, batch_coarse, dim_c)
        #obj_parser(fine_file, batch_fine, dim_f)
        obj2tri(coarse_file, batch_coarse)
        obj2tri(fine_file, batch_fine)

    # Trains the model for a fixed number of epochs (iterations on a dataset).
    x_train = np.array(batch_coarse) 
    y_list = []
    for i in range(len(batch_fine)):
        y_row = np.array(batch_fine[i]) - np.array(batch_coarse[i])
        y_list.append(y_row)
    y_train = np.array(y_list)
       
    print x_train.shape
    print y_train.shape
    # print x_train[0]
    # print y_train[0]
    
    # history = model.fit(x={'sim_input': x_train}, y={'smooth_out': y_train}, batch_size=32, epochs=20)
    history = model.fit(x_train, y_train)

    print(history.history.keys())

    # evaluate the model
    # scores = model.evaluate(x_train, y_train, batch_size=10)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # debug
    # print initial weigths
    # weights = model.layers[0].get_weights()
    # print weights
    

def save(model):
    # print_summary(model)
    model.save('my_model.h5') 
    model.save_weights('my_model_weights.h5')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", help="bool flag, False by default")
    parser.add_argument("--modelh5", help="load exist model")
    parser.add_argument("--modelweighth5", help="load model weights")
    args = parser.parse_args()
    # if args.load and args.modelh5:
    #     print "load pre-trained model..."
    #     model = load_model('my_model.h5')
    #     if args.modelweighth5:
    #         model.load_weights('my_model_weights.h5')
    # else:
        #batch_coarse = []
        #batch_fine = []
        #file_name = "00001_00.obj"
        #coarse_file = path_coarse + file_name
        #fine_file = path_tracking + file_name
        # obj_parser(coarse_file, fine_file, batch_coarse, batch_fine, dim)
        #setmodel()

    model = setmodel()

    train(model)

    save(model)


if __name__ == "__main__":
    main()


