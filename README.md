# Official CST-YOLO
<div style="display:flex;justify-content: center">
<a href="https://github.com/mkang315/CST-YOLO"><img src="https://img.shields.io/static/v1?label=GitHub&message=Code&color=black&logo=github"></a>
<a href="https://github.com/mkang315/CST-YOLO"><img alt="Build" src="https://img.shields.io/github/stars/mkang315/CST-YOLO"></a> 
<a href="https://huggingface.co/mkang315/CST-YOLO"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=yellow"></a>
<a href="https://arxiv.org/abs/2306.14590"><img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-2306.14590-b31b1b.svg"></a>
</div>

## Description
This is the source code for the paper titled "CST-YOLO: A Novel Method for Blood Cell Detection Based on Improved YOLOv7 and CNN-Swin Transformer" accepted by and presented orally at the 2024 IEEE International Conference on Image Processing ([ICIP 2024](https://2024.ieeeicip.org)), of which I am the first author. This paper is available to download from [IEEE Xplore](https://ieeexplore.ieee.org/document/10647618) or [arXiv](https://arxiv.org/abs/2306.14590).

## Model
The CNN-Swin Transformer You Only Look Once (CST-YOLO) model configuration (i.e., network construction) file is cst-yolo.yaml in the directory [./cfg/training/](https://github.com/mkang315/CST-YOLO/tree/main/cfg/training).

#### Installation
Install requirements.txt with recommended dependencies Python >= 3.8 environment including Torch <= 1.7.1 and CUDA <= 11.1:
```
pip install -r requirements.txt
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

## Referencing Guide
Please cite the paper if using this repository. Here is a guide to referencing this work in various styles for formatting your references:</br>

> Plain Text</br>
- **IEEE Reference Style**</br>
M. Kang, C.-M. Ting, F. F. Ting, and R. C.-W. Phan, "Cst-yolo: A novel method for blood cell detection based on improved yolov7 and cnn-swin transformer," in *Proc. IEEE Int. Conf. Image Process. (ICIP)*, Abu Dhabi, UAE, Oct. 27–30, 2024, pp. 3024–3029.</br>
<sup>**NOTE:** City of Conf., Abbrev. State, Country, Month & day(s) are optional.</sup>

- **IEEE Full Name Reference Style**</br>
Ming Kang, Chee-Ming Ting, Fung Fung Ting, and Raphaël C.-W. Phan. Cst-yolo: A novel method for blood cell detection based on improved yolov7 and cnn-swin transformer. In *ICIP*, pages 3024–3029, 2024.</br>
<sup>**NOTE:** This is a modification to the standard IEEE Reference Style and used by most IEEE/CVF conferences, including **CVPR**, **ICCV**, and **WACV**, to render first names in the bibliography as "Firstname Lastname" rather than "F. Lastname" or "Lastname, F.".</sup></br>
&nbsp;- **IJCAI Full Name-Year Variation**</br>
\[Kang *et al.*, 2024\] Ming Kang, Chee-Ming Ting, Fung Fung Ting, and Raphaël C.-W. Phan. Cst-yolo: A novel method for blood cell detection based on improved yolov7 and cnn-swin transformer. In *Proceedings of the 2024 IEEE International Conference on Image Processing*, pages 3024–3029, Piscataway, NJ, October 2024. IEEE.</br>
&nbsp;- **ACL Full Name-Year Variation**</br>
Ming Kang, Chee-Ming Ting, Fung Fung Ting, and Raphaël C.-W. Phan. 2024. Cst-yolo: A novel method for blood cell detection based on improved yolov7 and cnn-swin transformer. In *Proceedings of the 2024 IEEE International Conference on Image Processing*, pages 3024–3029, Piscataway, NJ. IEEE.</br>

- **Nature Referencing Style**</br>
Kang, M., Ting, C.-M., Ting, F. F. & Phan, R. C.-W. CST-YOLO: a novel method for blood cell detection based on improved YOLOv7 and CNN-swin Transformer. In *2024 IEEE International Conference on Image Processing (ICIP)* 3024–3029 (IEEE, 2024).</br>

- **Springer Reference Style**</br>
Kang, M., Ting, C.-M., Ting, F.F., Phan, R.C.-W.: CST-YOLO: a novel method for blood cell detection based on improved YOLOv7 and CNN-swin Transformer. In: 2024 IEEE International Conference on Image Processing (ICIP), pp. 3024–3029. IEEE, Piscataway (2024)</br>
<sup>**NOTE:** *ECCV* and *MICCAI* conference proceedings are part of the book series LNCS in which Springer's format for bibliographical references is strictly enforced. LNCS stands for Lecture Notes in Computer Science.</sup>

- **Elsevier Numbered Style**</br>
M. Kang, C.-M. Ting, F.F. Ting, R.C.-W. Phan, CST-YOLO: a novel method for blood cell detection based on improved YOLOv7 and CNN-swin Transformer, in: Proceedings of the IEEE International Conference on Image Processing (ICIP), 2024, pp. 3024–3029.</br>
<sup>**NOTE:** Day(s) Month Year, City, Abbrev. State, Country of Conference, Publiser, and Place of Publication are optional and omitted.</sup>

- **Elsevier Name–Date (Harvard) Style**</br>
Kang, M., Ting, C.-M., Ting, F.F., Phan, R.C.-W., 2024. CST-YOLO: a novel method for blood cell detection based on improved YOLOv7 and CNN-swin Transformer. In: Proceedings of the IEEE International Conference on Image Processing (ICIP), 27–30 October 2024, Abu Dhabi, UAE. IEEE, Piscataway, New York, USA, pp. 3024–3029.</br>
<sup>**NOTE:** Day(s) Month Year, City, Abbrev. State, Country of Conference, Publiser, and Place of Publication are optional.</sup>

- **Elsevier Vancouver Style**</br>
Kang M, Ting C-M, Ting FF, Phan RC-W. CST-YOLO: a novel method for blood cell detection based on improved YOLOv7 and CNN-swin Transformer. In: Proceedings of the IEEE International Conference on Image Processing (ICIP); 2024 Oct 27–30; Abu Dhabi, UAE. Piscataway: IEEE; 2024. p. 3024–9.</br>

- **Elsevier Embellished Vancouver Style**</br>
Kang M, Ting C-M, Ting FF, Phan RC-W. CST-YOLO: a novel method for blood cell detection based on improved YOLOv7 and CNN-swin Transformer. In: *Proceedings of the IEEE International Conference on Image Processing (ICIP)*; 2024 Oct 27–30; Abu Dhabi, UAE. Piscataway: IEEE; 2024. p. 3024–9.</br>

- **APA7 (Author–Date) Style**</br>
Kang, M., Ting, C.-M., Ting, F. F., & Phan, R. C.-W. (2024). CST-YOLO: A novel method for blood cell detection based on improved YOLOv7 and CNN-swin Transformer. In *Proceedings of the 2024 IEEE International Conference on Image Processing (ICIP)* (pp. 3024–3029). IEEE. https://doi.org/10.1109/ICIP51287.2024.10647618</br>
&nbsp;- **ICML (Author–Year) Variation**</br>
Kang, M., Ting, C.-M., Ting, F. F., and Phan, R. C.-W. CST-YOLO: A novel method for blood cell detection based on improved YOLOv7 and CNN-swin Transformer. In *Proceedings of the 2024 IEEE International Conference on Image Processing (ICIP)*, pp. 3024–3029, Piscataway, NJ, 2024. IEEE.</br>
<sup>**NOTE:** For **NeurIPS** and **ICLR**, any reference/citation style is acceptable as long as it is used consistently. The sample of references in Formatting Instructions For NeurIPS almost follows APA7 (author–date) style and that in Formatting Instructions For ICLR Conference Submissions is similar to IJCAI full name-year variation.</sup>

> BibTeX Format</br>
```
\begin{thebibliography}{1}
\bibitem{Kang24Cstyolo} M. Kang, C.-M. Ting, F. F. Ting, and R. C.-W. Phan, "Cst-yolo: A novel method for blood cell detection based on improved yolov7 and cnn-swin transformer," in {\emph Proc. IEEE Int. Conf. Image Process. (ICIP)}, Abu Dhabi, UAE, Oct. 27--30, 2024, pp. 3024--3029.
\end{thebibliography}
```
```
@inproceedings{Kang24Cstyolo,
  author = "Ming Kang and Chee-Ming Ting and Fung Fung Ting and Rapha{\"e}l C.-W. Phan",
  title = "Rcs-yolo: A fast and high-accuracy object detector for brain tumor detection",
  booktitle = "Proc. IEEE Int. Conf. Image Process. (ICIP)",
  % booktitle = ICIP, %% IEEE Full Name Reference Style
  address = "Abu Dhabi, UAE, Oct. 27--30",
  pages = "3024--3029",
  year = "2024"
}
```
```
@inproceedings{Kang24Cstyolo,
  author = "Kang, Ming and Ting, Chee-Ming and Ting, Fung Fung and Phan, Rapha{\"e}l C.-W.",
  title = "{CST-YOLO}: a novel method for blood cell detection based on improved {YOLO}v7 and {CNN}-swin transformer",
  editor = "",
  booktitle = "2024 IEEE International Conference on Image Processing (ICIP)",
  series = "",
  volume = "",
  pages = "3024--3029",
  publisher = "IEEE",
  address = "Piscataway",
  year = "2024",
  doi= "10.1109/ICIP51287.2024.10647618",
  url = "https://doi.org/10.1109/ICIP51287.2024.10647618"
}
```
<sup>**NOTE:** Please remove some optional *BibTeX* fields/tags such as `series`, `volume`, `address`, `url`, and so on if the *LaTeX* compiler produces an error. Author names may be manually modified if not automatically abbreviated by the compiler under the control of the bibliography/reference style (i.e., .bst) file. The *BibTex* citation key may be `bib1`, `b1`, or `ref1` when references appear in numbered style in which they are cited. The quotation mark pair `""` in the field could be replaced by the brace `{}`, whereas the brace `{}` in the *BibTeX* field/tag `title` plays a role of keeping letters/characters/text original lower/uppercases or sentence/capitalized cases unchanged while using Springer Nature bibliography style files, for example, sn-nature.bst.</sup>

## License
CST-YOLO is released under the GNU General Public License v3.0. Please see the [LICENSE](https://github.com/mkang315/CST-YOLO/blob/main/LICENSE) file for more information.

## Copyright Notice
Many utility codes of our project base on the codes of [YOLOv7](https://github.com/WongKinYiu/yolov7) and [Swin Transformer](https://github.com/microsoft/Swin-Transformer) repositories.
