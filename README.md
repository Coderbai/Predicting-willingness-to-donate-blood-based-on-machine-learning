# Predicting-willingness-to-donate-blood-based-on-machine-learning
A Keras implementation of Predicting willingness to donate blood based on machine learning: two blood donor recruitments during COVID-19 outbreaks

The article can be found here: https://www.nature.com/articles/s41598-022-21215-2

### Main codes of this project contains:

1. train_plot.py: plot models' learning curves based on certain data.
2. train_cv_10.py: train and test models to evaluate models' performance (accuracy, recall, precision, auc, f1)
3. process_data.py: process raw data to get what we need.
4. gen_data.py: get .npz files to be the input of models.
5. gen_label.py: get labels of the input.

> The data is not available due to the privacy concern.

Should you have any questions, you can write it in Issues.

### Citation
If you find this repository useful in your research, please consider citing the following paper:

```{
@article{wu2022predicting,
  title={Predicting willingness to donate blood based on machine learning: two blood donor recruitments during COVID-19 outbreaks},
  author={Wu, Hong-yun and Li, Zheng-gang and Sun, Xin-kai and Bai, Wei-min and Wang, An-di and Ma, Yu-chi and Diao, Ren-hua and Fan, Eng-yong and Zhao, Fang and Liu, Yun-qi and others},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={19165},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
