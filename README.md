# DeepDeconv 

#### descargar la data
Descargar la data de entrenamieto y todo desde:
```
https://figshare.com/articles/software/Deep_Deconvolution_for_Traffic_Analysis_with_Distributed_Acoustic_Sensing_Data/16653163
```

### Virtualenv en windows

installar virtualenv
```
pip install virtualenv
```

crear el env
```
virtualenv DeepDeconv
```
Activar el env usar
```
DeepDeconv\Scripts\activate
```

#### Instalar los requirements
```
pip install -r requirements.txt
```
### TEST MODEL
Para probar el modelo:
```
python test.py --weights "/weights/best.ckpt"
```

### TRAIN MODEL
Para entrenar el modelo:
```
python train.py --epochs 1000
```
