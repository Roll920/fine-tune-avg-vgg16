# coding=utf-8
import sys
caffe_root = '/home/luojh2/Software/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from PIL import Image


def im_resize(im, height=224, width=224):
    d_type = im.dtype
    im = Image.fromarray(im)
    im = im.resize([height, width], Image.BICUBIC)
    im = np.array(im, d_type)
    return im


def convert2rgb(im):
    if len(im.shape) == 2:
        im = im.reshape((im.shape[0], im.shape[1], 1))
        im = np.concatenate((im, im, im), axis=2)
    if im.shape[2] == 4:
        im = np.array(Image.fromarray(im).convert('RGB'))
    return im


def main():
    # 1. set parameters
    cub_path = '/opt/luojh/Dataset/CUB/images'
    model_weights = 'avg_vgg/snapshot/_iter_3948.caffemodel'
    model_def = 'avg_vgg/deploy.prototxt'
    gpu_device = 6

    # 2. load images list
    list_fp = open(cub_path + '/image_labels/val.txt', 'r')
    img_label_list = list_fp.readlines()
    img_list = [item.split()[0] for item in img_label_list]
    label_list = [int(item.split()[1]) for item in img_label_list]
    list_fp.close()

    # 2. load net
    caffe.set_device(gpu_device)
    caffe.set_mode_gpu()
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    mean_value = np.array([110, 127, 123], dtype=np.float32)
    mean_value = mean_value.reshape([3, 1, 1])

    # 3. predict
    num_tp = 0
    for img_index, img_name in enumerate(img_list):
        im = Image.open('%s/val/%s' % (cub_path, img_name))
        im = convert2rgb(np.array(im))
        im = im_resize(im, 256, 256)
        im = np.array(im, np.float64)
        im = im[:, :, ::-1]  # convert RGB to BGR
        im = im.transpose((2, 0, 1))  # convert to 3x256x256
        im -= mean_value
        # shape for input (data blob is N x C x H x W), set data
        # center crop
        im = im[:, 16:240, 16:240]
        net.blobs['data'].reshape(1, *im.shape)
        net.blobs['data'].data[...] = im
        # run net and take argmax for prediction
        net.forward()
        out = net.blobs['prob'].data[0]
        if np.argmax(out) == label_list[img_index]:
            num_tp += 1
        if img_index % 100 == 0:
            print '%s images done, %d left' % (img_index, len(img_list) - img_index)
    print 'acc: %f' % (float(num_tp) / len(img_list))


if __name__ == '__main__':
    main()
