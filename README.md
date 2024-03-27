# Official CST-YOLO
This is the source code for the paper, "CST-YOLO: A Novel Method for Blood Cell Detection Based on Improved YOLOv7 and CNN-Swin Transformer", of which I am the first author.

## Model
The model configuration (i.e., network construction) file is cst-yolo.yaml in the directory [./cfg/training/](https://github.com/mkang315/CST-YOLO/tree/main/cfg/training).

Recommended dependencies:
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
We trained and evaluated CST-YOLO on three blood cell detection datasets [Blood Cell Count and Detection (BCCD)](https://github.com/Shenggan/BCCD_Dataset), [Complete Blood Count (CBC)](https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset), and [Blood Cell Detection (BCD)](https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset). The 60 samples of the validation set duplicate those from the training set in the CBC dataset. Each image includes three types of blood cells: Red Blood Cells (RBCs), White Blood Cells (WBCs), and platelets.

**Table 1&nbsp;&nbsp;&nbsp;&nbsp;Number of examples in BCCD, CBC, and BCD.** 
| Dataset | Training | Validation | Testing | Total |
| :--------: | :-------: | :-------: | :-------: | :-------: |
| [BCCD](https://github.com/Shenggan/BCCD_Dataset) | 205 | 87 | 72 | 364 |
| [CBC](https://github.com/MahmudulAlam/Complete-Blood-Cell-Count-Dataset) | 300 | 0 | 60 | 360 |
| [BCD](https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset) | 255 | 73 | 36 | 364 |

<br /> 

**Table 2.1&nbsp;&nbsp;&nbsp;&nbsp;Performance comparison of YOLOv5x, YOLOv7 and CST-YOLO on the BCCD dataset. Results are mAP@0.5 for each blood cell
type and overall performance. The best results are shown in bold.** 
| Model | WBCs | RBCs | Platelets | Overall |
| :-------: | :-------: | :-------: | :-------: | :-------: |
| [RT-DETR-R50vd](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch) | — | — | — | 0.875 |
| [YOLOv5x](https://github.com/ultralytics/yolov5) | 0.977 | **0.877** | 0.915 | 0.923 |
| [YOLOv7](https://github.com/WongKinYiu/yolov7) | 0.977 | 0.829 | 0.883 | 0.896 |
| **CST-YOLO** | **0.984** | 0.869 | **0.928** | **0.927** |

**Table 2.2&nbsp;&nbsp;&nbsp;&nbsp;Performance comparison of YOLOv5x, YOLOv7 and CST-YOLO on the CBC dataset. Results are mAP@0.5 for each blood cell
type and overall performance. The best results are shown in bold.** 
| Model | WBCs | RBCs | Platelets | Overall |
| :-------: | :-------: | :-------: | :-------: | :-------: |
| [RT-DETR-R50vd](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch) | — | — | — | 0.855 |
| [YOLOv5x](https://github.com/ultralytics/yolov5) | 0.995 | 0.930 | **0.942** | 0.955 |
| [YOLOv7](https://github.com/WongKinYiu/yolov7) | 0.995 | 0.917 | 0.912 | 0.941 |
| **CST-YOLO** | 0.995 | **0.947** | 0.927 | **0.956** |

**Table 2.3&nbsp;&nbsp;&nbsp;&nbsp;Performance comparison of YOLOv5x, YOLOv7 and CST-YOLO on the BCD dataset. Results are mAP@0.5 for each blood cell
type and overall performance. The best results are shown in bold.** 
| Model | WBCs | RBCs | Platelets | Overall |
| :-------: | :-------: | :-------: | :-------: | :-------: |
| [RT-DETR-R50vd](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch) | — | — | — | 0.784 |
| [YOLOv5x](https://github.com/ultralytics/yolov5) | 0.820 | 0.857 | 0.975 | 0.884 |
| [YOLOv7](https://github.com/WongKinYiu/yolov7) | 0.874 | 0.785 | 0.974 | 0.878 |
| **CST-YOLO** | **0.899** | **0.857** | **0.978** | **0.911** |

## Ablation Study
The tables below show the effect on the performance of the proposed modules for the three blood cell datasets.

**Table 3.1&nbsp;&nbsp;&nbsp;&nbsp;Abation study of the proposed modules. Results are mAP@0.5 on the BCCD dataset. The best results are shown in bold.** 
| Method | WBCs | RBCs | Platelets| Overall |
| :-------: | :-------: | :-------: | :-------: | :-------: |
| w/o CST | 0.849 | 0.841 | 0.900 | 0.894 |
| w/o W-ELAN | 0.866 | 0.860 | 0.987 | 0.904 |
| w/o MCS | 0.762 | 1.000 | 0.987 | 0.910 |
| w/o MaxPool | 0.856 | 0.806 | 0.901 | 0.899 |

**Table 3.2&nbsp;&nbsp;&nbsp;&nbsp;Abation study of the proposed modules. Results are mAP@0.5 on the CBC dataset. The best results are shown in bold.** 
| Method | WBCs | RBCs | Platelets| Overall |
| :-------: | :-------: | :-------: | :-------: | :-------: |
| w/o CST | 0.999 | 0.944 | 0.923 | 0.955 |
| w/o W-ELAN | 0.995 | 0.954 | 0.910 | 0.953 |
| w/o MCS | 0.998 | 0.954 | 0.868 | 0.940 |
| w/o MaxPool | 0.999 | 0.920 | 0.938 | 0.952 |

**Table 3.3&nbsp;&nbsp;&nbsp;&nbsp;Abation study of the proposed modules. Results are mAP@0.5 on the BCD dataset. The best results are shown in bold.** 
| Method | WBCs | RBCs | Platelets| Overall |
| :-------: | :-------: | :-------: | :-------: | :-------: |
| w/o CST | 0.975 | 0.860 | 0.869 | 0.901 |
| w/o W-ELAN | 0.980 | 0.851 | 0.924 | 0.918 |
| w/o MCS | 0.973 | 0.878 | 0.857 | 0.903 |
| w/o MaxPool | 0.975 | 0.827 | 0.849 | 0.884 |

## Generalizability Study
Table 4 compares the performance between the original YOLOv7 and CST-YOLO on the [TinyPerson dataset](https://github.com/ucas-vg/PointTinyBenchmark/tree/TinyBenchmark) in different domain from medical images for external validation. The experimental results demonstrate the generalizability effectivenss of CST-YOLO in small object detection on both medical (i.e., blood cell) and natural (i.e., tiny person) images.

**Table 4&nbsp;&nbsp;&nbsp;&nbsp;Performance comparison of YOLOv7 and CST-YOLO for all classes of the TinyPerson dataset. The best results are shown in bold.**
| Model | Precision | Recall | mAP<sub>50</sub> | mAP<sub>50:95</sub> |
| :-------: | :-------: | :-------: | :-------: | :-------: |
| [YOLOv7](https://github.com/WongKinYiu/yolov7) | 0.530 | **0.361** | 0.335 | 0.111 |
| **CST-YOLO** | **0.565** | 0.358 | **0.344** | 0.111 |

## Visual Illustrations for Model Explanation
Some heat maps are generated by [Grad-CAM](https://github.com/ramprs/grad-cam) from the [BCCD](https://github.com/Shenggan/BCCD_Dataset) dataset. Examples of heat maps and failure cases are shown in Fig. 1 and Fig. 2, respectively.</br>
<p float="left">
<img src="https://github.com/mkang315/CST-YOLO/blob/main/heatmaps/BloodImage_00261/BloodImage_00261.jpg" alt="BloodImage_00261" width="200" align="middle" />
<img src="https://github.com/mkang315/CST-YOLO/blob/main/heatmaps/BloodImage_00261/Platelets.png" alt="Platelets" width="200" align="middle" />
<img src="https://github.com/mkang315/CST-YOLO/blob/main/heatmaps/BloodImage_00261/WBCs.png" alt="WBCs" width="200" align="middle" />
<img src="https://github.com/mkang315/CST-YOLO/blob/main/heatmaps/BloodImage_00261/RBCs.png" alt="RBCs" width="200" align="middle" />
<figcaption>(a) BloodImage_00261.jpg&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(b) Platelets&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(c) WBCs&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(d) RBCs</br> 
<b>Fig. 1 Example of Grad-CAM heat maps. Input the original blood cell image (a) BloodImage_00261.jpg to be detected and generated heat maps. The three heat maps with gradient-weighted class activation mapping respectively emphasize the objects in different classes: (b) Platelets, (c) WBCs, and (d) RBCs in the blood cell image, and de-emphasize the other two classes. Overall, we have a much more precise region of emphasis that locates the different types of blood cell. We know that the model classifies this input image due to its intrinsic features, not a general region in the image.</b></figcaption>
</p>

Below is one example of failure cases because there aren't any Grad-CAM heatmaps solely generated for platelets. The model ability of dim object detection need be enhanced for future improvements.
<p float="left">
<img src="https://github.com/mkang315/CST-YOLO/blob/main/heatmaps/BloodImage_00340/BloodImage_00340.jpg" alt="BloodImage_00340" width="200" align="middle" />
<img src="https://github.com/mkang315/CST-YOLO/blob/main/heatmaps/BloodImage_00340/WBCs.png" alt="WBCs" width="200" align="middle" />
<img src="https://github.com/mkang315/CST-YOLO/blob/main/heatmaps/BloodImage_00340/RBCs.png" alt="RBCs" width="200" align="middle" />
<figcaption>(a) BloodImage_00340.jpg&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(b) WBCs&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(c) RBCs</br> 
<b>Fig. 2 Example of failure cases. The heat maps of (b) WBCs, and (c) RBCs are generated through inputting the original blood cell image (a) BloodImage_00340.jpg.</b></figcaption>
</p>

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
