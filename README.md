# DeepDeconv

Deep learning based deconvolution algorithm implemented with a U-Net.

#### Download data
Download training data from:
```
https://figshare.com/articles/software/Deep_Deconvolution_for_Traffic_Analysis_with_Distributed_Acoustic_Sensing_Data/16653163
```

### Virtualenv on Windows

Install virtualenv
```
pip install virtualenv
```

Create environment
```
virtualenv DeepDeconv
```
Activate environment
```
DeepDeconv\Scripts\activate
```

#### Install requirements
```
pip install -r requirements.txt
```
### TEST MODEL
Testing the model:
```
python test.py --weights "/weights/best.ckpt"
```

### TRAIN MODEL
Training the model:
```
python train.py --epochs 1000
```
