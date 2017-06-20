'''
Image retrieval
'''
import os
import subprocess
import numpy as np
from sklearn.neighbors import KDTree

from layer_features import layer_features


def binary_hash_codes(feature_mat):
    """convert feature matrix of latent layer to binary hash codes"""
    xs, ys = np.where(feature_mat > 0.5)
    code_mat = np.zeros(feature_mat.shape)

    for i in range(len(xs)):
        code_mat[xs[i]][ys[i]] = 1

    return code_mat


def retrieve_image(target_image, model_file, deploy_file, imagemean_file,
                   threshold=1):
    model_dir = os.path.dirname(model_file)
    image_files = np.load(os.path.join(model_dir, 'image_files.npy'))
    fc7_feature_mat = np.load(os.path.join(model_dir, 'fc7_features.npy'))
    latent_feature_file = os.path.join(model_dir, 'latent_features.npy')
    latent_feature_mat = np.load(latent_feature_file)

    candidates = []
    dist = 0
    for layer, mat in layer_features(['latent', 'fc7'], model_file,
                                     deploy_file, imagemean_file,
                                     [target_image], show_pred=True):
        if layer == 'latent':
            # coarse-level search
            mat = binary_hash_codes(mat)
            mat = mat * np.ones((latent_feature_mat.shape[0], 1))
            dis_mat = np.abs(mat - latent_feature_mat)
            hamming_dis = np.sum(dis_mat, axis=1)
            distance_file = os.path.join(model_dir, 'hamming_dis.npy')
            np.save(distance_file, hamming_dis)
            candidates = np.where(hamming_dis < threshold)[0]

        if layer == 'fc7':
            # fine-level search
            kdt = KDTree(fc7_feature_mat[candidates], metric='euclidean')
            k = 6

            if not candidates.shape[0] > 6:
                k = candidates.shape[0]

            dist, idxs = kdt.query(mat, k=k)
            candidates = candidates[idxs]
            print(dist)

    return image_files[candidates][0], dist[0]


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5:
        usage = 'Usage: python retrieve.py' + \
                ' model_file deploy_file imagemean_file target_image.jpg'
        print(usage)
    else:
        model_file = sys.argv[1]
        deploy_file = sys.argv[2]
        imagemean_file = sys.argv[3]
        target_image = sys.argv[4]

        is_exists = os.path.exists(model_file) and os.path.exists(deploy_file)\
            and os.path.exists(imagemean_file)

        if is_exists:
            res, _ = retrieve_image(target_image, model_file, deploy_file,
                                    imagemean_file, threshold=5)
            print(res)
            if not os.path.exists('results'):
                os.mkdir('results')
            for i in range(len(res)):
                subprocess.call(['cp', res[i], 'results/%s.jpg' % str(i)])
        else:
            print('The model related files may not exit')
            print('Please check files: {}, {}, {}'
                  .format(model_file, deploy_file, imagemean_file))
