# coling18-gte





## Running the BiMPM models
### Prerequisite
1. Tensorflow (I've only tested version 1.0)
2. Flick30k + Keras and its [VGG16 pretrained models](https://github.com/fchollet/deep-learning-models) or download [feature file](bimpm/image_features/vgg_feats_77512_2.npy)
3. [Glove embeddings](https://nlp.stanford.edu/projects/glove/) (I used Glove but other embeddings should work)

### Train
File .config contains all the settings and hyperparameters for training. In order to run:
1. Obtain the image features by either extracting using Keras and Flickr30k dataset or download the features file (git-lfs)
   - To extract image features by yourself, specify location of Flickr30k dataset in [image_utils.py](bimpm/image_utils.py)
        and run ``` python image_utils.py```
2. Specify location of the embedding in the config file
3. ```python main.py --config_file=file_config_name_here.config```

Trained models are saved in [models directory](bimpm/models). If you want to run decode only, change `decoding_only` to `true`.
