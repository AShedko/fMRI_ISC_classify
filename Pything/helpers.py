import gzip
import pickle
from glob import glob

import nibabel as nib
import tensorflow as tf
from const import *


class Loader0:
    def __init__(self, path=PATH0):
        self.path = path
        self.fnames = dict()
        for i in range(1, 8):
            # print (i)
            self.fnames[i] = sorted(glob(self.path + '/' + str(i) + '/fmri/*.nii'))

    def get_img(self, subject, img):
        return nib.load(self.fnames[subject][img]).dataobj


class Loader:
    def __init__(self, path=PATH):
        self.path = path
        self.fnames = dict()
        for i in range(1, 30):
            print (i)
            self.fnames[i] = sorted(glob(self.path + '/' + str(i) + '/*.nii.gz'))

    def get_img(self, subject, img):
        return nib.load(self.fnames[subject][img]).dataobj

#---#
# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

def load_object_from_gzip_file(filenameWithPath) :
    f = gzip.open(filenameWithPath, 'rb')
    loaded_obj = pickle.load(f)
    f.close()
    return loaded_obj

def dump_object_to_gzip_file(my_obj, filenameWithPath) :
    f = gzip.open(filenameWithPath, 'wb')
    pickle.dump(my_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

#---#

def mean_4d(p):
    l = Loader()
    vol0 = l.get_img(p,0)
    acc = vol0.get_data()
    sess = tf.Session()
    x = tf.placeholder(tf.float32)
    a = tf.placeholder(tf.float32)
    ind = tf.placeholder(tf.float32)
    sl = (x*ind+a)/(ind+1)
    for i in range(1,3620):
        vol = l.get_img(p,i).get_data()
        acc = sess.run(sl,{x:acc,ind:i, a:vol})
        if i%400 ==0: print(i)
    im = nib.Nifti1Image(acc,vol0.affine)
    im.to_filename(l.path +str(p)+'/mean.nii')
