import argparse
import os, sys
import os.path
import operator
import numpy as np 
from keras import utils
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Lambda, merge, multiply
from keras.layers import Activation, Embedding
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from m_layers import Smooth, Densepool, compressmtx
from util import obj_parser, face2mtx, obj2tri, tri2obj, write_obj


def setmodel(input_shape, mtx):
    # 1 subdivide & pooling
    mesh_in = Input((input_shape[0],9), name='mesh_input')
    smoo = Smooth(activation='relu', name='smoo_layer')(mesh_in)
    pool = Densepool(mtx=mtx, output_dim=input_shape[1], name='pool_layer')(smoo)
    
    # model
    model = Model(inputs=[mesh_in], outputs=[pool])
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse', 'acc'])

    print "set model and compile"
    model.summary()

    return model


# extend dataset x y
def load_data(path_coarse, path_tracking):
    batch_coarse = []
    vert_coarse = []
    vert_fine = []
    vert_delta = []

    for x in xrange(1,101):
        file_name = str(x).zfill(5) + '_00.obj'
        coarse_file = path_coarse + '/' + file_name
        fine_file = path_tracking + '/' + file_name
        obj2tri(coarse_file, batch_coarse)
        # obj2tri(fine_file, batch_fine)
        obj_parser(coarse_file, vert_coarse)
        obj_parser(fine_file, vert_fine)
        
    # Trains the model for a fixed number of epochs (iterations on a dataset).
    x_train = np.array(batch_coarse) 
    
    for i in range(len(vert_fine)):
        y_row = np.array(vert_fine[i]) - np.array(vert_coarse[i])
        vert_delta.append(y_row)
    y_target = np.array(vert_delta)
  
    return x_train, y_target


def train(model, x_train, y_train):
    history = model.fit(x_train, y_train, batch_size=32, epochs=20)
    print(history.history.keys())


def eval(model, x_test, y_target):
    # evaluate the model
    scores = model.evaluate(x_test, y_target, batch_size=32)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Generates output predictions for the input samples.
def pred(x, v_dim, obj_in, obj_out):
    print "load pre-trained model..."
    model = load_model('my_model.h5')
    model.load_weights('my_model_weights.h5')
    y = model.predict(x, batch_size=32)
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
        sample_data = []
        file_name = coarseDir + [ f for f in os.listdir(coarseDir) if not f.startswith('.')][0] + "/00001_00.obj"
        dim = obj2tri(file_name, sample_data)   # [tri_dim, vert_dim]
        mtx = face2mtx(file_name, dim)
        # create model
        model = setmodel(dim, mtx)
        x_train = np.empty(0)
        y_train = np.empty(0)
        x_test = np.empty(0)
        y_test = np.empty(0)

        print ">>>>>>> loading data..."
        for dirName, subdirList, fileList in os.walk(coarseDir):
            total = len(subdirList)
            count = 0
            for subdir in subdirList:
                # print('Found directory: %s' % subdir)
                if count%5 == 0:
                    print str(float(count)/total*100) + '%'
                count = count + 1
                x, y = load_data(coarseDir + subdir, fineDir + subdir)
                if x_train.size == 0:
                    x_train = x
                    y_train = y
                else: 
                    x_train = np.vstack((x_train, x))
                    y_train = np.vstack((y_train, y))  

        if x_train.size == 0:
            print "Error: no input training data."
            return 0

        print ">>>>>>> train model..."
        print x_train.shape
        print y_train.shape
        train(model, x_train, y_train)

        print 'load test data to evaluate...'
        for dirName, subdirList, fileList in os.walk(test_coarseDir):
            for subdir in subdirList:
                print('Found directory: %s' % subdir)
                x, y = load_data(test_coarseDir + subdir, test_fineDir + subdir)
                if x_test.size == 0:
                    x_test = x
                    y_test = y
                else:
                    x_test = np.vstack((x_test, x))
                    y_test = np.vstack((y_test, y))

        if x_test.size == 0:
            print "Error: Need test dataset."
            return 0
        
        eval(model, x_test, y_test)

    save(model)


if __name__ == "__main__":
    main()


