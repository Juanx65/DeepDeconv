# DeepDeconv

Deep learning based deconvolution algorithm implemented with a U-Net.

Please see also: https://github.com/Juanx65/DeepDeconvV2

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

In case "cannot be loaded because running scripts is disabled on this system":
```
Set-ExecutionPolicy Unrestricted -Scope Process
```



# extras
If u get error on tensorflow not working with your gpu:
(this is the download link: https://www.dll-files.com/cudnn64_8.dll.html). When you get the file, you can just put in your bin directory. For example, usually in windows platform, you can put it into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin.
