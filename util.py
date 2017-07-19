import os, sys
import numpy as np
import math


def to_list(x):
    '''This normalizes a list/tensor into a list.
    If a tensor is passed, we return
    a list of size 1 containing the tensor.
    '''
    if type(x) is list:
        return x
    return [x]


# x y must be in same dimention
def custom_add(x, y):
    return x + y


# extend dataset x y
def load_data(path_coarse, path_tracking):
    batch_coarse = []
    batch_fine = []
    
    for x in xrange(1,101):
        file_name = str(x).zfill(5) + '_00.obj'
        coarse_file = path_coarse + '/' + file_name
        fine_file = path_tracking + '/' + file_name
        obj2tri(coarse_file, batch_coarse)
        obj2tri(fine_file, batch_fine)
        
    # Trains the model for a fixed number of epochs (iterations on a dataset).
    x_train = np.array(batch_coarse) 
    y_list = []
    for i in range(len(batch_fine)):
        y_row = np.array(batch_fine[i]) - np.array(batch_coarse[i])
        y_list.append(y_row)
    y_train = np.array(y_list)

    return x_train, y_train


# face index to index matrix
def face2mtx(objfile, dim):
    if not os.path.isfile(objfile):
        print "file not exist"
        return
    
    mtx = np.array([np.zeros(dim[0]*3) for item in range(dim[1])])
    count = np.zeros((dim[1], 1))
    # print ">>> mtx shape: "
    # print mtx.shape
    tri_id = 0

    with open(objfile, "r") as f1:
        for line in f1:
            s = line.strip().split(' ')
            if s[0] == 'f':
                id1 = int(s[1].strip().split('/')[0]) - 1  # index start at 0
                id2 = int(s[2].strip().split('/')[0]) - 1
                id3 = int(s[3].strip().split('/')[0]) - 1
                #TODO
                # set the weight as 1
                mtx[id1][tri_id * 3] = 1
                mtx[id2][tri_id*3+1] = 1
                mtx[id3][tri_id*3+2] = 1
                tri_id = tri_id + 1

                count[id1][0] += 1.0
                count[id2][0] += 1.0
                count[id3][0] += 1.0
    
    mtx_1 = mtx
    mtx = mtx_1 / count

    return mtx, mtx_1


# load data to (pos vel) 6N dimention
# or:  load data to (pos ) 3N dimention
def obj_parser(file_name, batch_data):
    # position v, velocity nv
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
            # elif s[0] == 'nv':
                # vert.extend(map(float, s[1:]))
            # if len(vert) == 6:
            if len(vert) == 3:
                sample.append(vert)
                vert = []

    batch_data.append(sample)


# parse data as (p1 p2 p3) 9*tri_num (position only)
def obj2tri(file_name, batch_data):
    if not os.path.isfile(file_name):
        print file_name + " file not exist"
        return
    if not file_name.endswith('.obj'):
        print file_name + " Wrong file format, expect obj."
        return

    vert = []
    data = []
    dim = [0, 0]
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
    dim[0] = len(data)
    dim[1] = len(vert)
    batch_data.append(data)
    return dim


# convert result from NN to obj
# input: pos1 pos2 pos3 --> tri (v9*N)
# v_dim: total vertices num
# obj_in: original obj, just to copy index information
# obj_out: output obj
def tri2obj(input, v_dim, obj_in, obj_out):
    if not os.path.isfile(obj_in):
        print "file not exist"
        return
    
    vertices = np.array([np.empty(3) for item in range(v_dim)])
    tri_id = 0
    faces = []

    # read from original obj
    with open(obj_in, "r") as f1:
        for line in f1:
            s = line.strip().split(' ')
            if s[0] == 'f':
                id1 = int(s[1].strip().split('/')[0]) - 1  # index start at 1
                id2 = int(s[2].strip().split('/')[0]) - 1
                id3 = int(s[3].strip().split('/')[0]) - 1
                vertices[id1] = input[tri_id][0:3]
                vertices[id2] = input[tri_id][3:6]
                vertices[id3] = input[tri_id][6:9]
                tri_id = tri_id + 1
                # copy the face info
                faces.append(line)

    # check
    # print len(faces)       #1292
    # print vertices.shape   #700,3
    if vertices.shape == 0:
        print "Error: vertices.shape=0"

    # write to new file
    with open(obj_out, "w+") as f2:
        for x in range(v_dim):
            line = "v {} {} {}\n".format(vertices[x][0], vertices[x][1], vertices[x][2])
            f2.write(line)
        for face in faces:
            f2.write(face)


# input is 3*n position
def write_obj(input, v_dim, obj_in, obj_out):
    if not os.path.isfile(obj_in):
        print "file not exist"
        return
    
    inputs = to_list(input)
    print inputs[0]
    faces = []

    # read from original obj
    with open(obj_in, "r") as f1:
        for line in f1:
            s = line.strip().split(' ')
            if s[0] == 'f':
                # copy the face info
                faces.append(line)

    # check
    print len(faces)

    # write to new file
    with open(obj_out, "w+") as f2:
        for x in range(vertices.shape):
            line = "{} {} {}\n".format(inputs[x][0], inputs[x][1], inputs[x][2])
            f2.write(line)
        for face in range(len(faces)):
            f2.write(face)


# Adapting the learning rate
# 1. Decrease the learning rate gradually based on the epoch.
# 2. Decrease the learning rate using punctuated large drops at specific epochs.
# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate



