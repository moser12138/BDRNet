# BDRNet
Tips: 

1. **ss** means single scale evaluation, **ssc** means single scale crop evaluation, **msf** means multi-scale evaluation with flip augment, and **mscf** means multi-scale crop evaluation with flip evaluation. The eval scales and crop size of multi-scales evaluation can be found in [configs](./configs/).

2. The fps is tested in different way from the paper. For more information, please see [here](./tensorrt).

3. The authors of bisenetv2 used cocostuff-10k, while I used cocostuff-123k(do not know how to say, just same 118k train and 5k val images as object detection). Thus the results maybe different from paper. 

4. The authors did not report results on ade20k, thus there is no official training settings, here I simply provide a "make it work" result. Maybe the results on ade20k can be boosted with better settings.

5. The model has a big variance, which means that the results of training for many times would vary within a relatively big margin. For example, if you train bisenetv2 on cityscapes for many times, you will observe that the result of **ss** evaluation of bisenetv2 varies between 73.1-75.1. 


## deploy trained models

1. tensorrt  
You can go to [tensorrt](./tensorrt) for details.  

2. ncnn  
You can go to [ncnn](./ncnn) for details.  

3. openvino  
You can go to [openvino](./openvino) for details.  

4. tis  
Triton Inference Server(TIS) provides a service solution of deployment. You can go to [tis](./tis) for details.


## platform

My platform is like this: 

* ubuntu 18.04
* nvidia Tesla T4 gpu, driver 450.80.02
* cuda 10.2/11.3
* cudnn 8
* miniconda python 3.8.8
* pytorch 1.11.0


## get start

With a pretrained weight, you can run inference on an single image like this: 

```
$ python tools/demo.py --config configs/bisenetv2_city.py --weight-path /path/to/your/weights.pth --img-path ./example.png
```

This would run inference on the image and save the result image to `./res.jpg`.  

Or you can run inference on a video like this:  
```
$ python tools/demo_video.py --config configs/bisenetv2_coco.py --weight-path res/model_final.pth --input ./video.mp4 --output res.mp4
```
This would generate segmentation file as `res.mp4`. If you want to read from camera, you can set `--input camera_id` rather than `input ./video.mp4`.   


## prepare dataset

1.cityscapes  

Register and download the dataset from the official [website](https://www.cityscapes-dataset.com/). Then decompress them into the `datasets/cityscapes` directory:  
```
$ mv /path/to/leftImg8bit_trainvaltest.zip datasets/cityscapes
$ mv /path/to/gtFine_trainvaltest.zip datasets/cityscapes
$ cd datasets/cityscapes
$ unzip leftImg8bit_trainvaltest.zip
$ unzip gtFine_trainvaltest.zip
```

2.cocostuff   

Download `train2017.zip`, `val2017.zip` and `stuffthingmaps_trainval2017.zip` split from official [website](https://cocodataset.org/#download). Then do as following:  
```
$ unzip train2017.zip
$ unzip val2017.zip
$ mv train2017/ /path/to/BiSeNet/datasets/coco/images
$ mv val2017/ /path/to/BiSeNet/datasets/coco/images

$ unzip stuffthingmaps_trainval2017.zip
$ mv train2017/ /path/to/BiSeNet/datasets/coco/labels
$ mv val2017/ /path/to/BiSeNet/datasets/coco/labels

$ cd /path/to/BiSeNet
$ python tools/gen_dataset_annos.py --dataset coco
```

3.ade20k

Download `ADEChallengeData2016.zip` from this [website](http://sceneparsing.csail.mit.edu/) and unzip it. Then we can move the uncompressed folders to `datasets/ade20k`, and generate the txt files with the script I prepared for you:  
```
$ unzip ADEChallengeData2016.zip
$ mv ADEChallengeData2016/images /path/to/BiSeNet/datasets/ade20k/
$ mv ADEChallengeData2016/annotations /path/to/BiSeNet/datasets/ade20k/
$ python tools/gen_dataset_annos.py --ade20k
```


4.custom dataset  

If you want to train on your own dataset, you should generate annotation files first with the format like this: 
```
munster_000002_000019_leftImg8bit.png,munster_000002_000019_gtFine_labelIds.png
frankfurt_000001_079206_leftImg8bit.png,frankfurt_000001_079206_gtFine_labelIds.png
...
```
Each line is a pair of training sample and ground truth image path, which are separated by a single comma `,`.   

I recommand you to check the information of your dataset with the script:  
```
$ python tools/check_dataset_info.py --im_root /path/to/your/data_root --im_anns /path/to/your/anno_file
```
This will print some of the information of your dataset.  

Then you need to change the field of `im_root` and `train/val_im_anns` in the config file. I prepared a demo config file for you named [`bisenet_customer.py`](./configs/customer.py). You can start from this conig file.


## train

Training commands I used to train the models can be found in [here](./dist_train.sh).

## finetune from trained model

You can also load the trained model weights and finetune from it, like this:
```
$ export CUDA_VISIBLE_DEVICES=0,1
$ torchrun --nproc_per_node=2 tools/train.py --finetune-from ./res/model_final.pth --config ./configs/bisenetv2_city.py # or bisenetv1
```


## eval pretrained models
You can also evaluate a trained model like this: 
```
$ python tools/evaluate.py --config configs/cityscapes.py --weight-path /path/to/your/weight.pth
```
