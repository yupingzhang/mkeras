import argparse
import os, sys
import os.path
import operator
import numpy as np 
from keras import backend as K
from keras import utils
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Lambda, merge, multiply, add
from keras.layers import Activation, Embedding
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from m_layers import Smooth, Densepool
from util import load_data, obj_parser, face2mtx, obj2tri, tri2obj, write_obj, custom_add


def setmodel(input_shape, mtx, mtx_1):
    # 1 subdivide & pooling
    mesh_in = Input((input_shape[0], 9), name='mesh_input')
    output = Smooth(units=3, name='smoo_layer_1')(mesh_in)       # activation='relu', 
    # output = Densepool(mtx=mtx, mtx_1=mtx_1, activation='softplus', name='pool_layer_1')(smoo_1)  # 

    # output = Smooth(units=9, name='smoo_layer_2')(smoo_1)
    # output = Densepool(mtx=mtx, mtx_1=mtx_1, activation='softplus', name='pool_layer_2')(smoo_2)
    
    # model
    model = Model(inputs=[mesh_in], outputs=[output])
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse', 'acc'])

    print "set model and compile"
    model.summary()

    return model


def train(model, x_train, y_train):
    print ">>>>>>> train model..."
    history = model.fit(x_train, y_train, batch_size=32, epochs=20)
    print(history.history.keys())


def eval(model, x_test, y_target):
    # evaluate the model
    scores = model.evaluate(x_test, y_target, batch_size=32)
    print("Evaluate: \n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Generates output predictions for the input samples.
def pred(model, x, v_dim, obj_in, obj_out):
    print ">>> Predict "+ obj_in + " >>> " + obj_out
    # model = load_model('my_model.h5')
    # model.load_weights('my_model_weights.h5')
    print v_dim
    print x.shape
    y = model.predict(x, batch_size=32)
    print y.shape
    y = y + x
    # save the output data
    for i in xrange(0,100):
        filename = str(i+1).zfill(5) + '_00.obj'
        tri2obj(y[i], v_dim, obj_in, obj_out + filename)


def save(model):
    model.save('my_model.h5') 
    model.save_weights('my_model_weights.h5')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', help="training: coarse dir")
    parser.add_argument('--f', help="training: fine scale with track dir")
    parser.add_argument('--tc', help="test dataset: coarse dir")
    parser.add_argument('--tf', help="test dataset: fine scale with track dir")
    parser.add_argument('--x', help="predict input dataset dir") 
    parser.add_argument('--o', help="predict output dir") 
    parser.add_argument("--resume", help="bool flag, False by default")
    parser.add_argument("--modelh5", help="load exist model")
    parser.add_argument("--modelweighth5", help="load model weights")
    args = parser.parse_args()
    if len(sys.argv) < 4:
        print "Usage: --c=coarse_dir --f=tracking_dir optional* --> use --help"
        return 0

    coarseDir = args.c
    fineDir = args.f
    test_coarseDir = args.c
    test_fineDir = args.f
    pred_dir = args.c
    out_dir = 'apredict/'
    if args.tc and args.tf:
        test_coarseDir = args.tc
        test_fineDir = args.tf
    if args.x:
        pred_dir = args.x
    if args.o:
        out_dir = args.o
        if not os.path.exists(out_dir):
                os.makedirs(out_dir)
    

    print "training dataset: "
    print ">>>  " + coarseDir + "  >>>  " + fineDir
    print "test dataset: "
    print ">>>  " + test_coarseDir + "  >>>  " + test_fineDir

    v_dim = 0
    if args.modelh5:
        print "load trained model..."
        model = load_model('my_model.h5')
        if args.modelweighth5:
            model.load_weights('my_model_weights.h5')
    else:
        # load any file to get triangle dimention
        sample_data = []
        file_name = coarseDir + [ f for f in os.listdir(coarseDir) if not f.startswith('.')][0] + "/00001_00.obj"
        dim = obj2tri(file_name, sample_data)   # [tri_dim, vert_dim]
        v_dim = dim[1]
        mtx, mtx_1 = face2mtx(file_name, dim)
        # create model
        model = setmodel(dim, mtx, mtx_1)
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

        train(model, x_train, y_train)

        # print 'load test data to evaluate...'
        # for dirName, subdirList, fileList in os.walk(test_coarseDir):
        #     for subdir in subdirList:
        #         print('Found directory: %s' % subdir)
        #         x, y = load_data(test_coarseDir + subdir, test_fineDir + subdir)
        #         if x_test.size == 0:
        #             x_test = x
        #             y_test = y
        #         else:
        #             x_test = np.vstack((x_test, x))
        #             y_test = np.vstack((y_test, y))

        # if x_test.size == 0:
        #     print "Error: Need test dataset."
        #     return 0
        
        # eval(model, x_test, y_test)

    save(model)

    print ">>> weights: >>>> "
    weights = model.layers[1].get_weights()
    print weights

    # predict and save output to obj
    
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
            pred(model, x, v_dim, obj_in, out_dir + subdir + '/')
    

if __name__ == "__main__":
    main()


