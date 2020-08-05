# DenseNet  

This repository is pytorch implementation of [DenseNet](https://arxiv.org/abs/1608.06993).


## Requirments  

- python >= 3.8
- pytorch >= 1.4
- torchvision >= 0.5
- pandas >= 1.0

## How to use

You can train DenseNet by following command. 

```python
python run.py
```

Training results are put in work/* (* is setting name).  
When you want to train DenseNet with different hyper parameters, edit setting.yml and run bellow. 

```python
# * means setting name, default is cifar10+BC_k12.
python run.py -S *
```

## License

This repository is [MIT-Licensed](https://github.com/taketakeseijin/DenseNet/blob/master/LICENSE).

## Original Paper  

G.Huang, Z.Liu, L.van der Maaten, K.Q.Weinberger. Densely Connected Convolutional Networks. IEEE Conference on Pattern Recognition and Computer Vision (CVPR), 2016.