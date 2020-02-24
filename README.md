# A implementation of MobileNetV3
This is a implementation of [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf) training on pytorch, and then I convert it to caffe.

I trained the network as the paper's discription at first. The top-1 accuracy can reach 67.59% for small mobile models, but the top-1 accuracy can only reach 72.45% for large mobile models. Then I changed the order of last average pooling layer and its next convolution layer, thus the top-1 accuracy can reach 75.34% for large mobile models matching the result of the paper, and the top-1 accuracy is 68.44% for small mobile models.


## Accuracy

|               | MAdds  | Params | Top-1 acc | Pretrained Model|
| -----------  | --------- | ---------- | --------- | --------- | --------- |
| Offical large | 219M   | 5.4M   | 75.2% | |
| Offical small | 66M    | 2.9M   | 67.4% | |
| Ours large_old    | 240M   | 5.4M   | 72.% | |
| Ours small_old    | 66M    | 2.9M   | 67.6% |
| Ours large    | 296M   | 5.4M   | 75.3% | [google drive](https://drive.google.com/open?id=1C6cP5DzmaVc_-loHVy6ivexXzakxPeLP) |
| Ours small    | 99M    | 2.9M   | 68.4% | [google drive](https://drive.google.com/open?id=1HORcVmbs3JvinpV4giPm1caZdJyXIxPP) |


## Inference and validation

```
python inference.py mobilenetv3_large
python inference.py mobilenetv3_small
```
