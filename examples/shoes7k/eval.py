'''
Evaluation

Find the top-5 similar images
'''
import os
import sys
import subprocess

from convert_shoes7k_data import get_images

sys.path.append('../..')
import config
from retrieve import retrieve_image

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
EVAL_PATH = os.path.join(SCRIPT_PATH, 'evaluation')

# define model params
model_file = 'shoes7k_model_with_latent_layer_iter_10000.caffemodel'
deploy_file = 'deploy_with_latent_layer.prototxt'
imagemean_file = 'shoes7k_mean.npy'
MODEL_FILE = os.path.join(SCRIPT_PATH, model_file)
DEPLOY_FILE = os.path.join(SCRIPT_PATH, deploy_file)
IMAGE_MEAN = os.path.join(SCRIPT_PATH, imagemean_file)


def retrieve_single_image(image_file, d_threshold):
    """retrive similar images and copy
    the retrieved images to evaluation folder"""
    retrieved, dist = retrieve_image(image_file, MODEL_FILE, DEPLOY_FILE,
                                     IMAGE_MEAN)
    if dist[-1] < d_threshold:
        # this is a image that has acceptable similar top-5 images
        print('Retrieved image ' + image_file)
        image_name = os.path.basename(image_file)
        image_name = image_name.split('.')[0]
        eval_res_dir = os.path.join(EVAL_PATH, image_name)
        if not os.path.exists(eval_res_dir):
            os.mkdir(eval_res_dir)
            image_id = 0
            for similar_img in retrieved:
                img_name = os.path.basename(similar_img)
                res_img = '_'.join([str(image_id), img_name])
                res_img = os.path.join(eval_res_dir, res_img)
                subprocess.call(['cp', similar_img, res_img])
                image_id += 1


def eval_shoes7k(d_threshold=3):
    """Evaluate through all positive images"""
    if not os.path.exists(EVAL_PATH):
        os.mkdir(EVAL_PATH)

    images = get_images(config.eg_shoes7k_pos_path)
    for image_file in images:
        retrieve_single_image(image_file, d_threshold)


if __name__ == '__main__':
    eval_shoes7k()
