#### descargar la data
Descargar la data de entrenamieto y todo de
`https://figshare.com/articles/software/Deep_Deconvolution_for_Traffic_Analysis_with_Distributed_Acoustic_Sensing_Data/16653163`

### para usar virtualenv en windows

installar virtualenv `pip install virtualenv`

crea el virtualenv `virtualenv DeepDeconv`

Para entrar a el `DeepDeconv\Scripts\activate`


Para probar la data:

```
python test.py --weights "/weights/best.ckpt"
```


Para entrenar

```
python train.py --epochs 1000
```
