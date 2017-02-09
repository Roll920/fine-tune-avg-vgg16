# fine-tune-avg-vgg16

remove fc layers, replace pool5 with 14x14 average pooling and softmax layer, then fine tune on CUB200.

###### excute ######
1. modify generate_prototxt.py to specify the path of caffe, lmdb
2. ./fine_tune.sh

###### details ######
1. lmdb: 256*256 cub200 (#train: 5994, #val:5794)
2. adopt random crop to generate 224*224 images when training and use center crop when test
3. learning rate of softmax is 10x larger
4. batch size:32, 188 iters/epoch, train 21 epoch
5. learning rate change 10^-3~10^-5 (decay 10x every 7 epoch)

###### results ######

