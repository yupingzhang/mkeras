import argparse
import os.path
import numpy as np 
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Activation, Embedding
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


dim = []
model = Sequential()
# path_index = '053/'
path_coarse = 'sim_coarse/053d/'
path_tracking = 'sim_tracking/053/'


# load data
def obj_parser(coarse_file, fine_file, batch_coarse, batch_fine, dim):
    # position v, velocity nv
    dim_c = 0 # input dimension 6*cN
    dim_f = 0 # output dimension 6*fN
    if not os.path.isfile(coarse_file):
        print "file not exist"
        return
    
    vert = []
    coarse = []
    fine = []
    with open(coarse_file, "r") as f1:
        for line in f1:
            s = line.strip().split(' ')
            if s[0] == 'v':
                vert.extend(map(float, s[1:]))
            elif s[0] == 'nv':
                vert.extend(map(float, s[1:]))
            if len(vert) == 6:
                dim_c = dim_c + 1
                coarse.append(vert)
                vert = []

    with open(fine_file, "r") as f2:
        for line in f2:
            s = line.strip().split(' ')
            if s[0] == 'v':
                vert.extend(map(float, s[1:]))
            elif s[0] == 'nv':
                vert.extend(map(float, s[1:]))
            if len(vert) == 6:
                dim_f = dim_f + 1
                fine.append(vert)
                vert = []

    dim.append(dim_c)
    dim.append(dim_f)
    batch_coarse.append(coarse)
    batch_fine.append(fine)


def setmodel(dim):
    # define model
    # set to global --> model = Sequential()
    model.add(Dense(units=64, input_shape=(dim[0],6)))
    model.add(Activation('relu'))
    model.add(Dense(units=6))
    model.add(Activation('relu')

    # Configures the model for training.
    # model.compile(loss='mean_squared_error', optimizer='sgd')
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    # model.compile(optimizer='rmsprop',
    #           loss='binary_crossentropy',
    #           metrics=['accuracy'])
    
    print "set model"


def train(dim):
    batch_coarse = []
    batch_fine = []
    dim = []

    for x in xrange(1,101):
        file_name = str(x).zfill(5) + '_00.obj'
        coarse_file = path_coarse + file_name
        fine_file = path_tracking + file_name  # + path_index 
        obj_parser(coarse_file, fine_file, batch_coarse, batch_fine, dim)


    # Trains the model for a fixed number of epochs (iterations on a dataset).
    # fit
    # x: training data, y: target data
    x_train = np.array(batch_coarse)
    y_train = np.array(batch_fine)
    # x_train = x.reshape(x.shape[0], x.shape[1], 6, 1)
    # y_train = y.reshape(y.shape[0], y.shape[1], 6, 1)

    model.fit(x_train, x_train, batch_size=100, epochs=20)

    # evaluate the model
    scores = model.evaluate(x_train, y_train,  batch_size=100)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # debug
    # print initial weigths
    weights = model.layers[0].get_weights()
    

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
        batch_coarse = []
        batch_fine = []
        file_name = "00001_00.obj"
        coarse_file = path_coarse + file_name
        fine_file = path_tracking + file_name
        obj_parser(coarse_file, fine_file, batch_coarse, batch_fine, dim)
        print dim
        setmodel(dim)

    train(dim)

    save()


if __name__ == "__main__":
    main()


