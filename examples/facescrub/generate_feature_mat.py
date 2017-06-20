'''
Generate feature matrix

Prepare files for image retrieval:
image_files.npy
fc7_features.npy
latent_features.npy
'''
import os
import sys
import numpy as np

from convert_facescrub_data import get_all_images

sys.path.append('../..')
import config
from layer_features import layer_features
from retrieve import binary_hash_codes

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def generate_feature_matrix(model_file, deploy_file, imagemean_file):
    """generate feature matrix of image dataset
    save the matrix as npy file"""
    image_files = get_all_images()
    np.random.shuffle(image_files)

    # feed the network and get feature vectors
    feature_mat = {'fc7': [], 'latent': []}

    batch = []
    batch_size = 0
    for image in image_files:
        batch.append(image)
        batch_size += 1

        if batch_size == 1000:
            for layer, mat in layer_features(feature_mat.keys(), model_file,
                                             deploy_file, imagemean_file,
                                             batch):
                if layer == 'latent':
                    mat = binary_hash_codes(mat)

                feature_mat[layer].extend(mat)

            batch = []
            batch_size = 0

    if batch_size > 0:
        for layer, mat in layer_features(feature_mat.keys(), model_file,
                                         deploy_file, imagemean_file, batch):
            if layer == 'latent':
                mat = binary_hash_codes(mat)

            feature_mat[layer].extend(mat)

    # save to npy files
    np.save(os.path.join(SCRIPT_PATH, 'image_files.npy'), image_files)
    for layer in feature_mat.keys():
        npy_file = os.path.join(SCRIPT_PATH, layer + '_features.npy')
        np.save(npy_file, np.array(feature_mat[layer]))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        usage = 'Usage: python generate_feature_mat.py' + \
                ' model_file deploy_file imagemean_file'
        print(usage)
    else:
        model_file = sys.argv[1]
        deploy_file = sys.argv[2]
        imagemean_file = sys.argv[3]

        is_exists = os.path.exists(model_file) and os.path.exists(deploy_file)\
            and os.path.exists(imagemean_file)

        if is_exists:
            generate_feature_matrix(model_file, deploy_file, imagemean_file)
