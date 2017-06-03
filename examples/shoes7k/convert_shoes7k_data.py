'''
Convert shoes7k dataset to train/test lmdb dataset
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
TRAIN_LMDB = os.path.join(SCRIPT_PATH, 'shoes7k_train_lmdb')
TEST_LMDB = os.path.join(SCRIPT_PATH, 'shoes7k_test_lmdb')


def get_images(path):
    """get images under path into a numpy array"""
    image_types = ['.jpg', '.png']
    images = [os.path.join(path, i) for i in os.listdir(path) if i[-4:]
              in image_types]
    return np.array(images)


def get_tr_te_images(ratio):
    """get training and test images"""
    pos_images = get_images(config.eg_shoes7k_pos_path)
    neg_images = get_images(config.eg_shoes7k_neg_path)
    np.random.shuffle(pos_images)
    np.random.shuffle(neg_images)

    pos_split = int(pos_images.shape[0] * ratio)
    pos_train_images = pos_images[:pos_split]
    pos_test_images = pos_images[pos_split:]
    pos_train_labels = np.ones(pos_train_images.shape[0]).astype('int')
    pos_test_labels = np.ones(pos_test_images.shape[0]).astype('int')

    neg_split = int(neg_images.shape[0] * ratio)
    neg_train_images = neg_images[:neg_split]
    neg_test_images = neg_images[neg_split:]
    neg_train_labels = np.zeros(neg_train_images.shape[0]).astype('int')
    neg_test_labels = np.zeros(neg_test_images.shape[0]).astype('int')

    train_images = np.concatenate((pos_train_images, neg_train_images))
    train_labels = np.concatenate((pos_train_labels, neg_train_labels))
    test_images = np.concatenate((pos_test_images, neg_test_images))
    test_labels = np.concatenate((pos_test_labels, neg_test_labels))

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


def convert_shoes7k_data(ratio=.8):
    """read shoes7k pos and neg images and convert to lmdb"""
    tr_images, tr_labels, te_images, te_labels = get_tr_te_images(ratio)
    save_to_lmdb(tr_images, tr_labels, TRAIN_LMDB)
    save_to_lmdb(te_images, te_labels, TEST_LMDB)


if __name__ == '__main__':
    convert_shoes7k_data()
