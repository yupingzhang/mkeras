import os, sys
import numpy as np
from numpy import linalg as LA
import math
from operator import sub
from preprocess import load_batch


# x y must be in same dimention
def custom_add(x, y):
    return x + y


# extend dataset x y
def load_data(path_coarse, path_tracking, tdelta=True):
    batch_coarse = []
    batch_fine = []
    
    for x in xrange(1,101):
        file_name = str(x).zfill(5) + '_00.obj'
        coarse_file = path_coarse + '/' + file_name
        fine_file = path_tracking + '/' + file_name
        # obj2tri(coarse_file, batch_coarse)
        # obj2tri(fine_file, batch_fine)
        load_batch(coarse_file, batch_coarse)
        load_batch(fine_file, batch_fine)

    print "load data:", len(batch_coarse)
    # Trains the model for a fixed number of epochs (iterations on a dataset).
    x_train = np.array(batch_coarse) 
    print x_train.shape
    if tdelta:
        print "target data: delta "
        y_list = []
        for i in range(len(batch_fine)):
            y_row = np.array(batch_fine[i]) - np.array(batch_coarse[i])
            y_list.append(y_row)
        y_train = np.array(y_list)
    else:
        print "target data: batch fine "
        y_train = np.array(batch_fine)
    
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
    # print "=== count ==========================================="
    # print count
    # print "=== mtx_1 ==========================================="
    # print mtx_1
    # print "=== mtx ==========================================="
    # print mtx

    return mtx, mtx_1


# load data to (pos vel) 6N dimention
# or:  load data to (pos) 3N dimention
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


# output:  position velocity cotangent length area
def obj2tri2(file_name, batch_data, addvel=True, addcot=True, adddist=False, addarea=False):
    if not os.path.isfile(file_name):
        print file_name + " file not exist"
        return
    if not file_name.endswith('.obj'):
        print file_name + " Wrong file format, expect obj."
        return

    vert = []
    vel = []   # velocity
    data = []
    dim = [0, 0]
    with open(file_name, "r") as f1:
        for line in f1:
            s = line.strip().split(' ')
            if s[0] == 'v':
                vert.append(map(float, s[1:]))
            elif s[0] == 'nv':
                vel.extend(map(float, s[1:]))
            elif s[0] == 'f':
                face = []
                id1 = int(s[1].strip().split('/')[0]) - 1  # index start at 1
                id2 = int(s[2].strip().split('/')[0]) - 1
                id3 = int(s[3].strip().split('/')[0]) - 1
                # position
                p1 = vert[id1]
                p2 = vert[id2]
                p3 = vert[id3]
                face.extend(p1)
                face.extend(p2)
                face.extend(p3)
                # velocity
                if addvel:
                    face.extend([vel[id1], vel[id2], vel[id3]])
                # edge vectors
                v1 = np.array(p3) - np.array(p2) # p3 .- p2           
                v2 = np.array(p1) - np.array(p3)
                v3 = np.array(p2) - np.array(p1)
                if adddist:
                    dist1 = LA.norm(v1)
                    dist2 = LA.norm(v2)
                    dist3 = LA.norm(v3)
                    face.extend([dist1, dist2, dist3])
                if addcot:
                    ca1 = np.dot(-v2, v3) / LA.norm(np.cross(-v2, v3))      # cot
                    ca2 = np.dot(v1, -v3) / LA.norm(np.cross(v1, -v3))
                    ca3 = np.dot(-v1, v2) / LA.norm(np.cross(-v1, v2))
                    face.extend([ca1, ca2, ca3])
                if addarea:
                    area = 0.5 * LA.norm(np.cross(v1, v2));         # triangle area
                    face.extend([area, area, area])
                #add current line
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
def tri2obj_out(input, v_dim, obj_in, obj_out):
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
                # if vertices[id1].size == 0:
                #     vertices[id1] = input[tri_id][0:3]
                # if vertices[id2].size == 0:
                #     vertices[id2] = input[tri_id][3:6]
                # if vertices[id3].size == 0:
                #     vertices[id3] = input[tri_id][6:9]

                tri_id = tri_id + 1
                # copy the face info
                faces.append(line)


    #============= TODO: change to store the indices instead of opening the file everytime
    

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

#===only get faces once, store the indices/faces info instead of opening the file everytime
def getfaces(obj_in, indices, faces):
    # read from original obj
    with open(obj_in, "r") as f1:
        for line in f1:
            s = line.strip().split(' ')
            if s[0] == 'f':
                id1 = int(s[1].strip().split('/')[0]) - 1  # index start at 1
                id2 = int(s[2].strip().split('/')[0]) - 1
                id3 = int(s[3].strip().split('/')[0]) - 1
                indices.append([id1, id2, id3])
                faces.append(line)


def tri2obj(input, v_dim, indices, faces, obj_out):
    vertices = np.array([np.empty(3) for item in range(v_dim)])
    tri_id = 0
    faces = []

    dim = len(indices)
    for i in xrange(0,dim):
        vertices[indices[i][0]] = input[tri_id][0:3]
        vertices[indices[i][1]] = input[tri_id][3:6]
        vertices[indices[i][2]] = input[tri_id][6:9]
        tri_id = tri_id + 1

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
def write_obj(inputs, v_dim, obj_in, obj_out):
    if not os.path.isfile(obj_in):
        print "file not exist"
        return
    
    faces = []
    # read from original obj
    with open(obj_in, "r") as f1:
        for line in f1:
            s = line.strip().split(' ')
            if s[0] == 'f':
                # copy the face info
                faces.append(line)

    # write to new file
    with open(obj_out, "w+") as f2:
        for x in range(0,700):
            line = "v {} {} {}\n".format(inputs[x][0], inputs[x][1], inputs[x][2])
            f2.write(line)
        for face in faces:
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



