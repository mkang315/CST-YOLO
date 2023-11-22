# Official CST-YOLO
This is the source code for the paper, "CST-YOLO: A Novel Method for Blood Cell Detection Based on Improved YOLOv7 and CNN-Swin Transformer", of which I am the first author.

## Model
The model configuration (i.e., network construction) file is cst-yolo.yaml in the directory [./cfg/training/](https://github.com/mkang315/CST-YOLO/tree/main/cfg/training).

Recommended running environment:
```
Python <= 3.8
Torch <= 1.7.1
CUDA <= 11.1
```

#### Training

The hyperparameter setting file is hyp.scratch.p5.yaml in the directory [./data/](https://github.com/mkang315/CST-YOLO/tree/main/data).

###### Single GPU training
```
python train.py --workers 8 --device 0 --batch-size 32 --data data/cbc.yaml --img 640 640 --cfg cfg/training/cst-yolo.yaml --weights '' --name cst-yolo --hyp data/hyp.scratch.p5.yaml
```

###### Multiple GPU training
```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/cbc.yaml --img 640 640 --cfg cfg/training/cst-yolo.yaml --weights '' --name rcs-yolo --hyp data/hyp.scratch.p5.yaml
```

#### Testing

```
python test.py --data data/cbc.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights runs/train/exp/weights/best.pt --name val
```

## Evaluation
We trained and evaluated CST-YOLO on three blood cell detection datasets [Blood Cell Count and Detection (BCCD)](https://github.com/Shenggan/BCCD_Dataset), [Complete Blood Count (CBC)](https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset), and [Blood Cell Detection (BCD)](https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset).

**Table 1&nbsp;&nbsp;&nbsp;&nbsp;Number of examples in BCCD, CBC, and BCD.** 
| Dataset | Training | Validation | Testing | Total |
| :--------: | :-------: | :-------: | :-------: | :-------: |
| [BCCD](https://github.com/Shenggan/BCCD_Dataset) | 327 | 0 | 37 | 364 |
| [CBC](https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset) | 300 | 0 | 60 | 360 |
| [BCD](https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset) | 255 | 73 | 36 | 364 |

<br />
**Table 2&nbsp;&nbsp;&nbsp;&nbsp;Performance comparison of YOLOv5x, YOLOv7 and CST-YOLO for blood cell detection. Results are APs for each blood cell
type and mAP@0.5 for overall performance. The best results are shown in bold.** 
| Dataset | Model | WBC | RBC | Platelets | Overall |
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| ^^ | [DETR](https://github.com/facebookresearch/detr) | 0. | 0. | 0. | 0. |
| :[BCCD](https://github.com/Shenggan/BCCD_Dataset): | [YOLOv5x](https://github.com/ultralytics/yolov5) | 0. | 0. | 0. | 0. |
| ^^| [YOLOv7](https://github.com/WongKinYiu/yolov7) | 0. | 0. | 0. | **0.729** |
| ^^ | **CST-YOLO** | **0.936** | **0.945** | **0.946** | **0.729** |

## Ablation Study
<br />
**Table 3&nbsp;&nbsp;&nbsp;&nbsp;Abation study of the proposed modules. Results are APs for each blood cell type and mAP@0.5 for overall performance. The best results are shown in bold.** 
| Dataset | Method | WBC | RBC | Platelets| Overall |
| :--------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| &nbsp; | w/o CST | 0 | 37 | 364 | 364 |
| [BCCD](https://github.com/Shenggan/BCCD_Dataset) | w/o W-ELAN | 0 | 37 | 364 | 364 |
| &nbsp; | w/o MCS | 0 | 37 | 364 | 364 |
| &nbsp; | w/o MaxPool | 0 | 37 | 364 | 364 |
| [CBC](https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset) | w/o W-ELAN | 0 | 60 | 360 | 364 |
| [BCD](https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset) | w/o W-ELAN | 73 | 36 | 364 | 364 |

## Visualization
The heat maps of some blood cell images in the [BCCD](https://github.com/Shenggan/BCCD_Dataset) dataset are shown below.
![Image](src)

## Suggested Citation
Our manuscript has been uploaded on [arXiv](https://arxiv.org/abs/2306.14590). Please cite our paper if you use code from this repository:
> Plain Text

- *IEEE* Style</br>
M. Kang, C.-M. Ting, F. F. Ting, and R. C.-W. Phan, "Cst-yolo: A novel method for blood cell detection based on improved yolov7 and cnn-swin transformer," arXiv:2306.14590 [cs.CV], Jun. 2023.</br>

- *Nature* Style</br>
Kang, M., Ting, C.-M., Ting, F. F. & Phan, R. C.-W. CST-YOLO: a novel method for blood cell detection based on improved YOLOv7 and CNN-swin transformer. Preprint at https://arxiv.org/abs/2306.14590 (2023).</br>

- *Springer* Style</br>
Kang, M., Ting, C.-M., Ting, F. F., Phan, R.C.-W.: CST-YOLO: a novel method for blood cell detection based on improved YOLOv7 and CNN-swin transformer. arXiv preprint [arXiv:2306.14590](https://arxiv.org/abs/2306.14590) (2023)</br>

## License
CST-YOLO is released under the GNU General Public License v3.0. Please see the [LICENSE](https://github.com/mkang315/CST-YOLO/blob/main/LICENSE) file for more information.

## Copyright Notice
Many utility codes of our project base on the codes of [YOLOv7](https://github.com/WongKinYiu/yolov7) and [Swin Transformer](https://github.com/microsoft/Swin-Transformer) repositories.
