'''
Extract FC7 4096 feature vector
'''
import numpy as np
import caffe


def feed_net(model_file, deploy_file, imagemean_file, image_files, show_pred):
    """feed network"""
    n_files = len(image_files)
    net = caffe.Net(deploy_file, model_file, caffe.TEST)

    # define transformer for preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(imagemean_file).mean(1).mean(1))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    net.blobs['data'].reshape(n_files, 3, 227, 227)

    idx = 0
    for image in image_files:
        try:
            im = caffe.io.load_image(image)
            transformed_im = transformer.preprocess('data', im)
            net.blobs['data'].data[idx, :, :, :] = transformed_im
            idx += 1
        except Exception:
            pass

    out = net.forward()
    if show_pred:
        print(out['prob'].argmax())
    return net


def layer_features(layers, model_file, deploy_file, imagemean_file,
                   image_files, gpu=True, gpu_id=0, show_pred=False):
    """extract features from various layers"""
    if gpu:
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

    net = feed_net(model_file, deploy_file, imagemean_file, image_files,
                   show_pred)

    #if type(layers) == str:
        #return net.blobs[layers].data

    for layer in layers:
        if layer not in net.blobs:
            raise TypeError('Invalid layer name: ' + layer)
        yield (layer, net.blobs[layer].data)
