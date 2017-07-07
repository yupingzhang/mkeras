# util.py
import numpy as np


def to_list(x):
    '''This normalizes a list/tensor into a list.
    If a tensor is passed, we return
    a list of size 1 containing the tensor.
    '''
    if type(x) is list:
        return x
    return [x]


def custom(x):
    #e = K.exp(x)
    #s = K.sum(e, axis=-1, keepdims=True)
    #return e / s
    return x


# face index to one-hot matrix
def face2mtx(objfile, f_dim):
    if not os.path.isfile(objfile):
        print "file not exist"
        return
    
    mtx = np.array([np.zeros(f_dim) for item in range(f_dim)])
    tri_id = 0

    with open(file_name, "r") as f1:
        for line in f1:
            s = line.strip().split(' ')
            if s[0] == 'f':
                id1 = int(s[1].strip().split('/')[0]) - 1  # index start at 1
                id2 = int(s[2].strip().split('/')[0]) - 1
                id3 = int(s[3].strip().split('/')[0]) - 1
                #TODO
                # set the weight as 1
                mtx[tri_id][id1] = 1
                mtx[tri_id][id2] = 1
                mtx[tri_id][id3] = 1
                tri_id = tri_id + 1

    return mtx


# load data to (pos vel) 6N dimention
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


# parse data as (p1 p2 p3) 9*tri_num (position only)
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
                # face.extend(vert[id1])
                # face.extend(vert[id2])
                # face.extend(vert[id3])
                # update, extend by index order
                idx = [id1, id2, id3].sort()
                face.extend(vert[idx[0]])
                face.extend(vert[idx[1]])
                face.extend(vert[idx[2]])
                data.append(face)
    batch_data.append(data)


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
                # put vertex position into 'vertices'
                print input[tri_id][0:3]
                vertices[id1] = input[tri_id][0:3]
                vertices[id2] = input[tri_id][3:6]
                vertices[id3] = input[tri_id][6:9]
                # copy the face info
                faces.append(line)

    # check
    print len(faces)
    print vertices.shape
    if vertices.shape == 0:
        print "Error: vertices.shape=0"

    # write to new file
    with open(obj_out, "w+") as f2:
        for x in range(vertices.shape):
            line = "{} {} {}\n".format(vertices[x][0], vertices[x][1], vertices[x][2])
            f2.write(line)
        for face in range(len(faces)):
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




