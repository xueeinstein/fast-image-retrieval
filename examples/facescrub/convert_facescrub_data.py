'''
Convert facescrub dataset to train/test lmdb dataset

------------------------------------------------------
Two classes: actor, actress

The dataset is downloaded using https://github.com/faceteam/facescrub
Please keep the folder structure after downloaded,
and configure the config.eg_facescrub_folder as path to `facescrub`
where `download.py` exists.
'''
import os
import cv2
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

import config


IM_HEIGHT = 227
IM_WIDTH = 227
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
TRAIN_LMDB = os.path.join(SCRIPT_PATH, 'facescrub_train_lmdb')
TEST_LMDB = os.path.join(SCRIPT_PATH, 'facescrub_test_lmdb')
ACTORS = os.path.join(config.eg_facescrub_folder, 'facescrub_actors.txt')
ACTRESS = os.path.join(config.eg_facescrub_folder, 'facescrub_actresses.txt')
DOWNLOAD = os.path.join(config.eg_facescrub_folder, 'download')


def get_names(path):
    """get actor or actress names"""
    data = np.loadtxt(path, delimiter='\t', skiprows=1, dtype=str)
    return np.unique(data[:, 0])


def get_images(names, ratio, label, train_images, train_labels,
               test_images, test_labels):
    for name in names:
        folder = '_'.join(name.split())
        folder = os.path.join(DOWNLOAD, folder, 'face')

        faces = os.listdir(folder)
        split = int(len(faces) * ratio)
        for idx, face in enumerate(faces):
            face = os.path.join(folder, face)
            if idx < split:
                train_images.append(face)
                train_labels.append(label)
            else:
                test_images.append(face)
                test_labels.append(label)


def get_tr_te_images(ratio):
    """get training and test images for two classes"""
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    actors = get_names(ACTORS)
    actresses = get_names(ACTRESS)

    get_images(actors, ratio, 1, train_images, train_labels, test_images,
               test_labels)
    get_images(actresses, ratio, 0, train_images, train_labels, test_images,
               test_labels)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # shuffle
    train_idxs = np.arange(train_images.shape[0])
    test_idxs = np.arange(test_images.shape[0])
    np.random.shuffle(train_idxs)
    np.random.shuffle(test_idxs)

    return (train_images[train_idxs], train_labels[train_idxs],
            test_images[test_idxs], test_labels[test_idxs])


def save_to_lmdb(images, labels, lmdb_file):
    if not os.path.exists(lmdb_file):
        batch_size = 256
        lmdb_env = lmdb.open(lmdb_file, map_size=int(1e12))
        lmdb_txn = lmdb_env.begin(write=True)
        item_id = 0
        datum = caffe_pb2.Datum()

        for i in range(images.shape[0]):
            im = cv2.imread(images[i])
            if im is None:
                continue
            im = cv2.resize(im, (IM_HEIGHT, IM_WIDTH))
            datum.channels = im.shape[2]
            datum.height = im.shape[0]
            datum.width = im.shape[1]
            datum.data = im.tobytes()
            datum.label = labels[i]
            keystr = '{:0>8d}'.format(item_id)
            lmdb_txn.put(keystr, datum.SerializeToString())

            # write batch
            if (item_id + 1) % batch_size == 0:
                lmdb_txn.commit()
                lmdb_txn = lmdb_env.begin(write=True)
                print('converted {} images'.format(item_id + 1))

            item_id += 1

        # write last batch
        if (item_id + 1) % batch_size != 0:
            lmdb_txn.commit()
            print('converted {} images'.format(item_id + 1))
            print('Generated ' + lmdb_file)
    else:
        print(lmdb_file + ' already exists')


def convert_facecrub_data(ratio=.8):
    tr_images, tr_labels, te_images, te_labels = get_tr_te_images(ratio)
    save_to_lmdb(tr_images, tr_labels, TRAIN_LMDB)
    save_to_lmdb(te_images, te_labels, TEST_LMDB)


if __name__ == '__main__':
    convert_facecrub_data()
