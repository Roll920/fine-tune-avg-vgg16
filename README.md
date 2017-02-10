# Fine tune avg-vgg16 on CUB200 dataset
Remove fc layers, replace pool5 with 14x14 average pooling and softmax layer, then fine tune on CUB200.

## How to use
* modify generate_prototxt.py to specify the path of caffe, lmdb
* ./fine_tune.sh

## Implementation details
* lmdb: 256*256 cub200 dataset(#train: 5994, #val:5794).
* adopt random cropping to generate 224*224 images when training and use center cropping when test.
* training issues:
	* learning rate of softmax is 10x larger;
	* batch size:32, 188 iters/epoch, train 21 epoch;
	* use 2 gpu cards;
	* learning rate change 10^-3~10^-5 (decay 10x every 7 epoch);

## Results
| epoch | val acc |
| -- | --|

## Others
This strategy is much better than train softmax first follwed with fine tuning, which only achieving 0.70176 accuracy!

