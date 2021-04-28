# Py-dnn
Deep Neural Networks Models powered by Pytorch

**version v0.0.1**

### Plans

- [x] ImageClassifier (CNN model)
  
  > Transfer learning (using well-known models like **ResNet**)

- [ ] TextClassifier (CNN and RNN model)
   
  > Using word embeddings downloaded from Tencent or Sogou
  
- [ ] TextWriter 

   maybe writing poems?


### Modules

```shell
├── LICENSE
├── README.md
├── common.py
├── criterion
│   └── criterion.py
├── data_loader
│   └── image_loader.py
├── evaluate
│   └── image_classify_eval.py
├── image_classify_run.py
├── model
│   ├── image_classify_model.py
│   └── save.py
├── optimizer
│   └── optimizer.py
├── options
│   ├── base_opt.py
│   └── image_classify_opt.py
├── plot
│   └── plot_writer.py
└── resources
    ├── data
    ├── images
    ├── log
    ├── results
    └── saved_model
```

### Usages

**like this**
```shell
$ usage: image_classify_run.py [--data DATA] [--out OUT] [--log LOG]
                             [--model_save MODEL_SAVE]
                             [--model {VGG,ResNet,DenseNet,ResNext}] [--again]
                             [--test] [--batch_size BATCH_SIZE]
                             [--epoch_num EPOCH_NUM] [--optimizer {Adam}]
                             [--learning_rate LEARNING_RATE] [--beta1 BETA1]
                             [--beta2 BETA2]
                             [--lr_scheduler {StepLR,ExponentialLR,CosineAnnealingLR}]
                             [--gamma GAMMA] [--loss {NLLLoss}]
                             [--thread_num THREAD_NUM] [--gpu_ids GPU_IDS]
                             [--plot] [--seed SEED] [--help]
```

| Resource Arguments | Dest       | Help                                                                                                                                                                           |
|--------------------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --data             | DATA       | for train, the path should have sub-folders train and valid;for test, should have sub-folders test. [reference docs](https://pytorch.org/vision/stable/datasets.html#imagefolder) |
| --log              | LOG        | path to the log folder to record information.                                                                                                                                  |
| --model_save       | MODEL_SAVE | models are saved here.
| etc..      | etc.. | etc..

**for example**

```shell
$ python image_classify_run.py --plot \
                               --model ResNet \
                               --data resources/data/flowers \
                               --model_save resources/saved_model/my_ResNet.pth \
                               --lr_scheduler StepLR \
                               --again
```

### Logs 

**for example**

```shell
$ tensorboard.exe --logdir .\log\
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.4.1 at http://localhost:6006/ (Press CTRL+C to quit)
```
**like this**

![img1](./resources/images/1.png)
![img2](./resources/images/2.png)


### Cites

`# todo`