# Chinese Character (Traditional) Scene Text

<div style="text-align: justify">

This code is used for the competition [繁體中文場景文字辨識競賽－進階賽：繁體中文場景文字辨識](https://tbrain.trendmicro.com.tw/Competitions/Details/16). The competition is about scene text recognition in Taiwan (Traditional Chinese). This code assembled [YOLOv5](https://github.com/ultralytics/yolov5) and [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) in one. YOLOv5 for text detection and EfficientNet-PyTorch for Chinese character image classification. The dataset for text detection included [Google Landmarks Dataset V2](https://github.com/cvdfoundation/google-landmark) as background images while for image classification included [Single_char_image_generator](https://github.com/rachellin0105/Single_char_image_generator) as character generator. Both the dataset also included the data given by the organiser. 

</div>

---
<div style="text-align: justify">
<font size = "3">

<b>NOTE :exclamation:</b> <br>
This code <b>IS NOT</b> the latest version of YOLOv5. Therefore, training using this code will not compatible with the code provided on [YOLOv5](https://github.com/ultralytics/yolov5), or vice versa.

</font>
</div>

---

## Qucik Start
### Installation
[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/RabbitCrab/Chinese-Character-Traditional-Scene-Text/blob/main/requirements.txt) installed including

```
$ git clone https://github.com/RabbitCrab/Chinese-Character-Traditional-Scene-Text.git
$ cd Chinese-Character-Traditional-Scene-Text
$ pip install -r requirements.txt
```

### Inference
Please download the YOLOv5 weight ([final_weight.pt](https://drive.google.com/file/d/1HLKZAnQrFpJbp3NOgxTPf5UHFSIl-_O_/view?usp=sharing)) and EfficientNet model ([eff_final_imgsz64.pt](https://drive.google.com/file/d/1oCd0Xz2BGgIoNq8sBoElnIwwrDE-fkE0/view?usp=sharing)). <br>
Place the EfficientNet model under the [resnet_training](https://github.com/RabbitCrab/Chinese-Character-Traditional-Scene-Text/tree/main/resnet_training)

```
Project
|---README.md
|---final_test.py
|---final_weight.pt # Not neccessary need to follow
|---data
|   |---public
|   |   |---img_public
|   |   |---Task2_Public_String_Coordinate.csv
|   |---private
|   |   |---img_private
|   |   |---Task2_Private_String_Coordinate.csv
|---resnet_training
|   |---train_efficientnet.py
|   |---eff_final_imgsz64.pt
```

<div style="text-align: justify">

<br> `final_test.py` runs inference on a `path/` (directory) and output the result in `*.csv` <br>
The `final_test.py` is designed to run for the competition [繁體中文場景文字辨識競賽－進階賽：繁體中文場景文字辨識](https://tbrain.trendmicro.com.tw/Competitions/Details/16). <br>
Hence, several parts in the code requires to change accordingly. <br>
Line 33: the path for `task.csv` that needs to read in. <br>
Line 306: you probably need to change the output file name. <br>
The public dataset from the competition provided can be download from [HERE](https://drive.google.com/file/d/1E09fzyjJLAtciDi7fInn-CsJyhVevhUw/view?usp=sharing). <br><br>
Run the following command:

</div>

```
python final_test.py --img 1280 --weight final_weight.pt --augment --source data/public/img_public
```

When finished, the result will be saved and output as `output_final.csv`.


## Dataset
There are two datasets used for the YOLOv5 training and EfficientNet training. <br>
Dataset: <br>
[Dataset for YOLOv5](https://drive.google.com/file/d/1awQpoOd7GkdA6FyPxbNKXRScdynWC5fI/view?usp=sharing) <br>
[EfficientNet_images](https://drive.google.com/file/d/1HSrA5a20zAuGtIgHfpCaN3NXwKMddWmd/view?usp=sharing) <br>
[EfficientNet_labels](https://drive.google.com/file/d/12ER-I_WiymmxqtjEQ3WKbH8SZfZV22kk/view?usp=sharing) <br>


## Training
### YOLOv5
Place the dataset for YOLOv5 as following:

```
|---datasets
|   |---contest
|   |   |---annotations
|   |   |   |---train.txt
|   |   |   |---val.txt
|   |   |   |---text.yaml
|   |   |---images
|   |   |   |---train
|   |   |   |   |---img_1.jpg
|   |   |   |---val
|   |   |   |   |---img_3653.jpg
|   |   |---labels
|   |   |   |---train
|   |   |   |   |---img_1.txt
|   |   |   |---val
|   |   |   |   |---img_3653.txt
|---Project

```
Download YOLOv5 model from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) <br>
Train with the following command:

```
# Single GPU
python train.py --img 1280 --weight yolov5x6.pt --data ../datasets/contest/annotations/text.yaml --batch-size 16 # Use the largest batch size as GPU allows
```

#### Custom Dataset

<div style="text-align: justify">

Please follow the [documentation](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) and run the above command. <br>
**Again, take note that the latest YOLOv5 is not compatible with this code.** <br>
Note: This code used Google Landmarks Dataset v2 as background images. Images are mainly from western countries to prevent contradictions for Chinese characters.

</div>  

### EfficientNet

```
Project
|---resnet_training
|   |---train_efficientnet.py
|   |---eff_final_imgsz64.pt
|   |---crop_image_new
|   |   |---img_1.jpg
|   |---crop_label_new
|   |   |---img_1.txt
```

Train with the following command:

```
cd resnet_training
python train_efficientnet.py
```

### Custom Dataset

<div style="text-align: justify">

1. Create your own mapping relation between the characters and numbers in JSON file. Example: [new_num_to_word.json](https://github.com/RabbitCrab/Chinese-Character-Traditional-Scene-Text/blob/main/resnet_training/new_num_to_word.json)
2. Create label for each of the image.
3. Create `*.txt` to locate the image paths for training. Example: [new_crop_total.txt](https://github.com/RabbitCrab/Chinese-Character-Traditional-Scene-Text/blob/main/resnet_training/new_crop_total.txt)
4. Change line 22 in [train_efficientnet.py](https://github.com/RabbitCrab/Chinese-Character-Traditional-Scene-Text/blob/main/resnet_training/train_efficientnet.py) to your folder name.
5. Change line 79 in [train_efficientnet.py](https://github.com/RabbitCrab/Chinese-Character-Traditional-Scene-Text/blob/main/resnet_training/train_efficientnet.py) to your number of class(es).
6. Change line 89 in [train_efficientnet.py](https://github.com/RabbitCrab/Chinese-Character-Traditional-Scene-Text/blob/main/resnet_training/train_efficientnet.py) to your
relation JSON file.

Note: You may use [Single_char_image_generator](https://github.com/rachellin0105/Single_char_image_generator) to generate more character image.

</div>


## Reference
1. [YOLOv5](https://github.com/ultralytics/yolov5)
2. [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
3. [Single_char_image_generator](https://github.com/rachellin0105/Single_char_image_generator)
4. [Google Landmarks Dataset V2](https://github.com/cvdfoundation/google-landmark)


## Contributions

