# README
## 简介
对MobileInst: Video Instance Segmentation on the Mobile论文中提出实例分割方法进行复现。本论文为视频实例分割任务，由于在VOC数据集上进行训练和测试，只实现图像实例分割部分。模型主要分为两个部分，backbone部分采用TopFormer模型，分割部分由两部分组成：Semantic-enhanced Mask Decoder和Dual Transformer Instance Decoder。

通过运行sbin中的脚本，可以进行对应实验，进行训练和测试。

## 结果
最终的COCO测试集结果如下，与论文仅相差2个点左右：
![](assets/final.png)
