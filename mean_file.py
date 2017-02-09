import sys
caffe_root = '/home/luojh2/Software/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np

blob = caffe.proto.caffe_pb2.BlobProto()
data = open('/opt/luojh/Dataset/CUB/LMDB/image_mean.binaryproto', 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
mean_value = arr[0]

print np.shape(mean_value)
print mean_value.mean(1).mean(1)
