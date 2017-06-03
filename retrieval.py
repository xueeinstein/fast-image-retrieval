'''
Image retrieval
'''
import numpy as np
from sklearn.neighbors import KDTree

from convert_shoes7k_data import get_images
from layer_features import layer_features
import config


def binary_hash_codes(feature_mat):
    """convert feature matrix of latent layer to binary hash codes"""
    xs, ys = np.where(feature_mat > 0.5)
    code_mat = np.zeros(feature_mat.shape)

    for i in range(len(xs)):
        code_mat[xs[i]][ys[i]] = 1

    return code_mat


def generate_feature_matrix(model_file, deploy_file, imagemean_file):
    """generate feature matrix of image dataset
    save the matrix as npy file"""
    # get image files
    pos_images = get_images(config.eg_shoes7k_pos_path)
    neg_images = get_images(config.eg_shoes7k_neg_path)

    image_files = np.concatenate((pos_images, neg_images))
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
    np.save('image_files.npy', image_files)
    for layer in feature_mat.keys():
        np.save(layer + '_features.npy', np.array(feature_mat[layer]))


def retrieve_image(target_image, model_file, deploy_file, imagemean_file, threshold=1):
    image_files = np.load('image_files.npy')
    fc7_feature_mat = np.load('fc7_features.npy')
    latent_feature_mat = np.load('latent_features.npy')

    candidates = []
    for layer, mat in layer_features(['latent', 'fc7'], model_file,
                                     deploy_file, imagemean_file,
                                     [target_image], show_pred=True):
        if layer == 'latent':
            # coarse-level search
            mat = binary_hash_codes(mat)
            mat = mat * np.ones((latent_feature_mat.shape[0], 1))
            dis_mat = np.abs(mat - latent_feature_mat)
            hamming_dis = np.sum(dis_mat, axis=1)
            np.save('hamming_dis.npy', hamming_dis)
            candidates = np.where(hamming_dis < threshold)[0]

        if layer == 'fc7':
            # fine-level search
            kdt = KDTree(fc7_feature_mat[candidates], metric='euclidean')
            k = 6
            if candidates.shape[0] > 6:
                dist, idxs = kdt.query(mat, k=k)
                candidates = candidates[idxs]
                print(dist)

    return image_files[candidates]


if __name__ == '__main__':
    import sys
    import os
    if len(sys.argv) != 5:
        print('Usage: python retrieval.py model_file deploy_file imagemean_file target_image')
    else:
        model_file = sys.argv[1]
        deploy_file = sys.argv[2]
        imagemean_file = sys.argv[3]
        target_image = sys.argv[4]

        if not (os.path.exists('image_files.npy') and os.path.exists('latent_features.npy') and os.path.exists('fc7_features.npy')):
            generate_feature_matrix(model_file, deploy_file, imagemean_file)

        res = retrieve_image(target_image, model_file, deploy_file, imagemean_file, threshold=5)
        print(res)
