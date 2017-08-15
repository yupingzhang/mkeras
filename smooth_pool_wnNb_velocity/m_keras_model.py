# import matplotlib as mpl
# mpl.use('Agg')

import argparse
import os, sys
import os.path
import operator
import numpy as np 
import logging
# import matplotlib.pyplot as plt
from keras import backend as K
from keras import utils
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Lambda, merge, multiply, add
from keras.layers import Activation, Embedding
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.optimizers import SGD
from m_layers import Smooth, Densepool
from util import load_data, obj_parser, face2mtx, obj2tri, tri2obj, write_obj, custom_add, step_decay, getfaces
from preprocess import meshmtx_wnb

nNb = 72  # 12 vert w/ pos+vel

def setmodel(input_shape, mtx, mtx_1):
    print input_shape[0]
    mesh_in = Input((input_shape[0], nNb), name='mesh_input')

    smoo_1 = Smooth(units=nNb, name='smoo_layer_1')(mesh_in)
    pool_1 = Densepool(mtx=mtx, mtx_1=mtx_1, name='pool_layer_1')(smoo_1)

    smoo_1 = Smooth(units=nNb, name='smoo_layer_2')(pool_1)
    pool_2 = Densepool(mtx=mtx, mtx_1=mtx_1, name='pool_layer_2')(smoo_1)

    smoo_1 = Smooth(units=nNb, name='smoo_layer_3')(pool_2)
    output = Densepool(mtx=mtx, mtx_1=mtx_1, name='pool_layer_3')(smoo_1)

    model = Model(inputs=[mesh_in], outputs=[output])
    model.summary()
    return model


def train(model, x_train, y_train, learning_rate, epochs, png):
    decay_rate = learning_rate / epochs
    sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=0.8, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse'])

    print ">>>>>>> train model >>>>>>> "
    lrate = LearningRateScheduler(step_decay)
    estop = EarlyStopping(monitor='mean_squared_error', min_delta=0.000001, patience=10, mode='min')
    history = model.fit(x_train, y_train, validation_split=0.20, batch_size=32, epochs=epochs, callbacks=[lrate, estop])

    print(history.history.keys())
    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(png)


def eval(model, x_test, y_target):
    # evaluate the model
    scores = model.evaluate(x_test, y_target, batch_size=32)
    print("Evaluate: \n%s: %.2f%%" % (model.metrics_names[0], scores[1]*100))


# Generates output predictions for the input samples.
def pred(model, x, v_dim, obj_in, obj_out):
    print ">>> Predict "+ obj_in + " >>> " + obj_out
    y = model.predict(x)
    # y = y + x
    # save the output data
    indices = []
    faces = []
    getfaces(obj_in, indices, faces)
    for i in xrange(0,100): 
        filename = str(i+1).zfill(5) + '_00.obj'
        tri2obj(y[i], v_dim, indices, faces, obj_out + filename)


def save(model):
    model.save('kmodel/my_model.h5') 
    model.save_weights('kmodel/my_model_weights.h5')
    del model
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action='store_true', default=False, help="train flag")
    parser.add_argument('-eval', action='store_true', default=False, help="evaluate flag")
    parser.add_argument('-pred', action='store_true', default=False, help="predict flag")
    parser.add_argument('-w', action='store_true', default=False, help="load weights flag")
    parser.add_argument('-c', help="training: coarse dir")
    parser.add_argument('-f', help="training: fine scale with track dir")
    parser.add_argument('-tc', help="test dataset: coarse dir")
    parser.add_argument('-tf', help="test dataset: fine scale with track dir")
    parser.add_argument('-x', help="predict input dataset dir") 
    parser.add_argument('-o', help="predict output dir") 
    parser.add_argument('-l', help="learning rate") 
    parser.add_argument('-e', help="epochs") 
    parser.add_argument('-p', help="png file name")
    parser.add_argument("-resume", help="bool flag, False by default")
    parser.add_argument("-modelh5", help="load exist model")
    parser.add_argument("-modelweighth5", help="load model weights")
    args = parser.parse_args()
    if len(sys.argv) < 4:
        print "Usage: --train=True -l=learning_rate -e=epochs -c=... -f=... --eval=False --pred=True option* --> use --help"
        return 0

    coarseDir = None
    fineDir = None
    test_coarseDir = None
    test_fineDir = None
    pred_dir = None
    out_dir = None
    png = "foo.png"

    if args.train:
        learning_rate = float(args.l)
        epochs = int(args.e)
        coarseDir = args.c
        fineDir = args.f
        png = args.p
        print "training dataset: "
        print ">>>  " + str(coarseDir) + "  >>>  " + str(fineDir)
    if args.eval:
        test_coarseDir = args.tc
        test_fineDir = args.tf
        print "evaluate dataset: "
        print ">>>  " + str(test_coarseDir) + "  >>>  " + str(test_fineDir)
    if args.pred:
        pred_dir = args.x
        out_dir = args.o
        if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        print "predict: "
        print ">>>  " + str(pred_dir) + "  >>>  " + str(out_dir)


    if coarseDir: sdir = coarseDir
    elif test_coarseDir: sdir = test_coarseDir
    elif pred_dir: sdir = pred_dir
    file_name = sdir + [f for f in os.listdir(sdir) if not f.startswith('.')][0] + "/00001_00.obj"
    # file_name = pred_dir + "test.obj"
    dim, mtx, mtx_1 = meshmtx_wnb(file_name)
    print "dim", dim
    v_dim = dim[1]
    # create model
    model = setmodel(dim, mtx, mtx_1)

    ##load predefined weights
    load_weights = args.w
    # if load_weights:
    #     alpha = 0.6
    #     beta = 0.2
    #     a1 = [alpha, 0.0, 0.0, beta, 0.0, 0.0, beta, 0.0, 0.0]
    #     a2 = [0.0, alpha, 0.0, 0.0, beta, 0.0, 0.0, beta, 0.0]
    #     a3 = [0.0, 0.0, alpha, 0.0, 0.0, beta, 0.0, 0.0, beta]
    #     a4 = [beta, 0.0, 0.0, alpha, 0.0, 0.0, beta, 0.0, 0.0]
    #     a5 = [0.0, beta, 0.0, 0.0, alpha, 0.0, 0.0, beta, 0.0]
    #     a6 = [0.0, 0.0, beta, 0.0, 0.0, alpha, 0.0, 0.0, beta]
    #     a7 = [beta, 0.0, 0.0, beta, 0.0, 0.0, alpha, 0.0, 0.0]
    #     a8 = [0.0, beta, 0.0, 0.0, beta, 0.0, 0.0, alpha, 0.0]
    #     a9 = [0.0, 0.0, beta, 0.0, 0.0, beta, 0.0, 0.0, alpha]
    #
    #     w = np.array([[a1, a2, a3, a4, a5, a6, a7, a8, a9]]) # has to be 1x(9x9) dim
    #     print ">>> predefined weights: "
    #     print w
    #     model.layers[1].set_weights(w)

    if args.train:    
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

        train(model, x_train, y_train, learning_rate, epochs, png)

    if args.eval:
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

    print ">>> weights: >>>> "
    weights = model.layers[1].get_weights()
    w1 = np.array(weights).astype(np.float32)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    print weights

    ## predict and save output to obj
    if args.pred:
        for dirName, subdirList, fileList in os.walk(pred_dir):
            for subdir in subdirList:
                newpath = out_dir + subdir
                print newpath
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                obj_in = pred_dir + subdir + '/00001_00.obj'
                batch_coarse = []
                for dirpath, dirnames, filenames in os.walk(pred_dir + subdir):
                    for x in xrange(1,101):
                        file_name = str(x).zfill(5) + '_00.obj'
                        obj2tri(pred_dir + subdir + '/' + file_name, batch_coarse)
                        
                x = np.array(batch_coarse)   
                # print "predict input: \n >>>>  "
                # print x.shape
                pred(model, x, v_dim, obj_in, out_dir + subdir + '/')

    # ============= test ==============
    # obj_in = pred_dir + "test.obj" 
    # batch_coarse = []
    # obj2tri(obj_in, batch_coarse)
                        
    # x = np.array(batch_coarse)   
    # pred(model, x, v_dim, obj_in, out_dir)
    # ============= test ==============

    save(model)
    

if __name__ == "__main__":
    main()


