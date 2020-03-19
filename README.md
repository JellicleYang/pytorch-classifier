# pytorch-classifier
  An idiot way to do the classification
  
  requirements:
  --python>=3
  --pytorch>=1.3.1

## Dataset
```
data
  --train
      --class1
      --class2
      ...
  --valid
      --class1
      --class2
      ...
``` 

## Train
```
sh run.sh
python train.py --trainroot data/train\
                --valroot data/valid\
                --num_classes 10\
                --batch_size 64\
                --gpu 0
```
## Convert model
  Model training is based on the DataParallel operation, but test operation needn't it. And we can also change the model which is trained on gpu to opreate on cpu. 

## Test
On cpu
```
python test_on_cpu
```
On gpu

```
python test_on_gpu
```

