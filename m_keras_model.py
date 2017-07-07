import argparse
import os, sys
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
from util import face2mtx, obj2tri, tri2obj


def setmodel(input_dim, mtx):
    # 1 model_1 subdivide & pooling
    # mesh_in = Input((input_dim, 9))
    # smoo = Smooth(input_shape=(input_dim,9), activation='relu', name='smoo_layer')(mesh_in) 
    model_1 = Sequential()
    model_1.add(Smooth(input_shape=(input_dim,9), activation='relu', name='smoo_layer'))
    model_1.add(Densepool(kernel_initializer=pooling_kernal_init(mtx)))
    
    # 2 model_2 
    mode_2 = Sequential()
    model_2.add(Lambda(lambda face_mtx: [1.0/float(x) for x in K.sum(face_mtx, axis=1)]))

    # merges
    merged = Merge([model_1, model_2], mode='multiply')

    # model
    # model = Model(inputs=[mesh_in, face_mtx], outputs=[output])
    model = Sequential()
    model.add(merged)
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse', 'acc'])

    print "set model and compile"
    model.summary()

    return model


# extend dataset x y
def load_data(path_coarse, path_tracking, x, y):
    batch_coarse = []
    batch_fine = []
    bath_delta = []

    for x in xrange(1,101):
        file_name = str(x).zfill(5) + '_00.obj'
        coarse_file = path_coarse + file_name
        fine_file = path_tracking + file_name
        obj2tri(coarse_file, batch_coarse)
        obj2tri(fine_file, batch_fine)

    # Trains the model for a fixed number of epochs (iterations on a dataset).
    x_train = np.array(batch_coarse) 
    y_list = []
    for i in range(len(batch_fine)):
        y_row = np.array(batch_fine[i]) - np.array(batch_coarse[i])
        y_list.append(y_row)
    y_train = np.array(y_list)

    if len(x) == 0 and len(y) == 0:
        x = x_train
        y = y_train
    else:
        x.extend(x_train)
        y.extend(y_train)


def train(model, x_train, mtx, y_train):
    history = model.fit('mesh_in': x_train, 'face_mtx': mtx, 'output': y_train, batch_size=32, epochs=20)
    print(history.history.keys())


def eval(model, x_test, mtx_test, y_target):
    # evaluate the model
    scores = model.evaluate([x_test, mtx_test], y_target, batch_size=32)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Generates output predictions for the input samples.
def pred(x, v_dim, obj_in, obj_out):
    print "load pre-trained model..."
    model = load_model('my_model.h5')
    model.load_weights('my_model_weights.h5')
    y = model.predict([x, mtx], batch_size=32)
    # save the output data
    tri2obj(y, v_dim, obj_in, obj_out)


def save(model):
    model.save('my_model.h5') 
    model.save_weights('my_model_weights.h5')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', help="training: coarse dir")
    parser.add_argument('--f', help="training: fine scale with track dir")
    parser.add_argument('--tc', help="test dataset: coarse dir")
    parser.add_argument('--tf', help="test dataset: fine scale with track dir")
    parser.add_argument("--resume", help="bool flag, False by default")
    parser.add_argument("--modelh5", help="load exist model")
    parser.add_argument("--modelweighth5", help="load model weights")
    args = parser.parse_args()
    if len(sys.argv) < 4:
        print "Usage: --c=coarse_dir --f=tracking_dir --tc=test_coarse --tf=test_tracking optional* --> use --help"
        return 0

    coarseDir = args.c
    fineDir = args.f
    test_coarseDir = args.tc
    test_fineDir = args.tf
    print type(coarseDir)
    print "training dataset: "
    print ">>>  " + coarseDir + "  >>>  " + fineDir
    print "test dataset: "
    print ">>>  " + test_coarseDir + "  >>>  " + test_fineDir

    if args.resume and args.modelh5:
        print "resume trained model..."
        model = load_model('my_model.h5')
        if args.modelweighth5:
            model.load_weights('my_model_weights.h5')
    else:
        # load any file to get triangle dimention
        batch_data = []
        file_name = os.listdir(coarseDir)[0] + "00000_00.obj"
        obj2tri(file_name, batch_data)
        dim = len(batch_data[0])
        mtx = face2mtx(objfile, f_dim)
        # debug
        print len(mtx)
        print len(mtx[0])

        # create model
        model = setmodel(dim, mtx)
        x_train = np.empty(0)
        y_train = np.empty(0)
        x_test = np.empty(0)
        y_test = np.empty(0)

        for dirName, subdirList, fileList in os.walk(coarseDir):
            print('Found directory: %s' % dirName)
            load_data(coarseDir + dirName, fineDir + dirName, x_train, y_train)

        print len(x)
        print x[0]

        train(model, x_train, y_train)

        for dirName, subdirList, fileList in os.walk(test_coarseDir):
            print('Found directory: %s' % dirName)
            load_data(test_coarseDir + dirName, test_fineDir + dirName, x_test, y_test)
        eval(model, x_test, y_test)

    save(model)


if __name__ == "__main__":
    main()


