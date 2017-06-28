import argparse
import os.path
import numpy as np 
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Activation, Embedding
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import m_layers


dim = []
model = Sequential()
# path_index = '053/'
path_coarse = 'sim_coarse/053d/'
path_tracking = 'sim_tracking/053/'


# load data
def obj_parser(file_name, batch_data, dim):
    # position v, velocity nv
    dim = 0 
    if not os.path.isfile(coarse_file):
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
    if not os.path.isfile(coarse_file):
        print "file not exist"
        return
    vert = []
    face = []
    with open(file_name, "r") as f1:
        for line in f1:
            s = line.strip().split(' ')
            if s[0] == 'v':
                vert.append(map(float, s[1:]))
            #elif s[0] == 'nv':
            #    vert.extend(map(float, s[1:]))
            elif s[0] == 'f':
                id1 = s[1].strip().split('/')[0] - 1  # index start at 1
                id2 = s[2].strip().split('/')[0] - 1
                id3 = s[3].strip().split('/')[0] - 1
                face.extend(vert[id1]).extend(vert[id2]).extend(vert[id3])
                batch_data.append(face)


def setmodel():
    # define model
    # set to global --> model = Sequential()
    #model.add(Dense(units=64, input_shape=(dim[0],6)))
        # kernel_regularizer=regularizers.l2(0.01),
        # activity_regularizer=regularizers.l1(0.01)))

    #todo
    model.add(Smooth(input_shape=(9,)))
    model.add(Activation('relu'))
    model.add(Dense(units=6))
    model.add(Activation('relu'))

    # Configures the model for training.
    # model.compile(loss='mean_squared_error', optimizer='sgd')
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    
    print "set model"


def train():
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
    y_train = np.array(batch_fine - batch_coarse)

    model.fit(x_train, x_train, batch_size=10, epochs=50)

    # evaluate the model
    scores = model.evaluate(x_train, y_train, batch_size=10)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # debug
    # print initial weigths
    weights = model.layers[0].get_weights()
    print weights
    

def save():
    model.save('my_model.h5') 
    model.save_weights('my_model_weights.h5')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelh5", help="load exist model")
    parser.add_argument("--modelweighth5", help="load model weights")
    args = parser.parse_args()
    if args.modelh5:
        model = load_model('my_model.h5')
        if args.modelweighth5:
            model.load_weights('my_model_weights.h5')
    else:
        #batch_coarse = []
        #batch_fine = []
        #file_name = "00001_00.obj"
        #coarse_file = path_coarse + file_name
        #fine_file = path_tracking + file_name
        # obj_parser(coarse_file, fine_file, batch_coarse, batch_fine, dim)
        setmodel()

    train()

    save()


if __name__ == "__main__":
    main()


