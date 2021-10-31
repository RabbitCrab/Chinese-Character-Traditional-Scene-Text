# Chinese Character (Traditional) Scene Text

This code is used for the competition [繁體中文場景文字辨識競賽－進階賽：繁體中文場景文字辨識](https://tbrain.trendmicro.com.tw/Competitions/Details/16). The competition is about scene text recognition in Taiwan (Traditional Chinese). This code included [YOLOv5](https://github.com/ultralytics/yolov5) for text detection and [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) for Chinese character image classification. The dataset for text detection included [Google Landmarks Dataset V2](https://github.com/cvdfoundation/google-landmark) as background image while for image classification included [Single_char_image_generator](https://github.com/rachellin0105/Single_char_image_generator) as character generator. Both the dataset also included the data given by the organiser. 

---
<font size = "3">

<b>NOTE :exclamation:</b> <br>
This code <b>IS NOT</b> the latest version of YOLOv5. Therefore, training using this code will not compatible with the code provided on [YOLOv5](https://github.com/ultralytics/yolov5).

</font>

---

## Qucik Start
### Installation
[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/RabbitCrab/Chinese-Character-Traditional-Scene-Text/blob/main/requirements.txt) installed including

```bash
$ git clone https://github.com/RabbitCrab/Chinese-Character-Traditional-Scene-Text.git
$ cd Chinese-Character-Traditional-Scene-Text
$ pip install -r requirements.txt
```

### Inference
Please download the YOLOv5 weight ([final_weight.pt](https://drive.google.com/file/d/1HLKZAnQrFpJbp3NOgxTPf5UHFSIl-_O_/view?usp=sharing)) and EfficientNet model ([eff_final_imgsz64.pt](https://drive.google.com/file/d/1oCd0Xz2BGgIoNq8sBoElnIwwrDE-fkE0/view?usp=sharing)). <br>
Place the EfficientNet model under the [resnet_training](https://github.com/RabbitCrab/Chinese-Character-Traditional-Scene-Text/tree/main/resnet_training)

```
Project
|---READ.md
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

<br> `final_test.py` runs inference on a `path/` (directory) and output the result in `*.csv` <br>
The `final_test.py` is designed to run for the competition [繁體中文場景文字辨識競賽－進階賽：繁體中文場景文字辨識](https://tbrain.trendmicro.com.tw/Competitions/Details/16). <br>
Hence, several parts in the code requires to change accordingly. <br>
Line 33: the path for `task.csv` that needs to read in. <br>
Line 306: you probably need to change the output file name. <br>
The public data can be download from [HERE](https://drive.google.com/file/d/1YzbssB91aEOBRS7iGWOtBoCZKbLHAhJo/view?usp=sharing). <br><br>
Run the following command:

```
python final_test.py --img 1280 --weight final_weight.pt --augment --source data/public/img
```

When finished, the result will be saved and output as `output_final.csv`.


## Dataset
There are two datasets used for the YOLOv5 training and EfficientNet training. <br>
Dataset: <br>
[YOLOv5]() <br>
[EfficientNet_images]() <br>
[EfficientNet_labels]() <br>


## Training


## Citation
