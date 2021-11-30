# Structured Multimodal Attentions for TextVQA

This repository contains the code for SMA model from the following paper, released under the Pythia framework:

Gao, C., Zhu, Q., Wang, P., Li, H., Liu, Y., Hengel, A. V. D., & Wu, Q. (2021). Structured multimodal attentions for textvqa. Accepted by TPAMI. ([PDF](https://arxiv.org/abs/2006.00753))

## Code
Please find our code in ./code. 

Code of the main structure of SMA 
```
./code/pythia/models/sma.py
```

Config Files 
```
./code/configs/vqa/textvqa/sma.yml
```

To setup the environment and train a model, please refer the installation step as [M4C](https://github.com/ronghanghu/mmf/tree/project/m4c_captioner_pre_release/projects/M4C)

## SBD-Trans OCR
The imdb files (with SBD-Trans OCR) for TextVQA dataset.

Download imdb files from [Google Drive Link](https://drive.google.com/drive/folders/1mMLgxHIf9Ev2W8OvXFyXodpDOHqVXrcf?usp=sharing)

## GT OCR Annotations 
The imdb files (with GT OCR labels) for TextVQA dataset.

Download imdb files from [Google Drive Link](https://drive.google.com/drive/folders/1wqdB6eIkz5DXRb0CIcCoNKCUmvksv5Ls?usp=sharing)

## Citation
If you find this project useful for your research, please cite:
```
@article{gao2021structured,
  title={Structured Multimodal Attentions for TextVQA},
  author={Gao, Chenyu and Zhu, Qi and Wang, Peng and Li, Hui and Liu, Yuliang and Hengel, Anton van den and Wu, Qi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021}
}
```
```
@inproceedings{SBD,
  title={Omnidirectional Scene Text Detection with Sequential-free Box Discretization},
  author={Yuliang Liu and Sheng Zhang and Lianwen Jin and Lele Xie and Yaqiang Wu and Zhepeng Wang},
  booktitle=IJCAI,
  pages={3052--3058},
  year={2019}
}
```
```
@article{yang2020holistic,
  title={A Holistic Representation Guided Attention Network for Scene Text Recognition},
  author={Yang, Lu and Wang, Peng and Li, Hui and Li, Zhen and Zhang, Yanning},
  journal={Neurocomputing},
  year={2020},
  publisher={Elsevier}
}
```
