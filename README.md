# Grounded Textual Entailment
Dataset and code associated to the paper [1].

## Abstract
Capturing semantic relations between sentences, such as entailment, is a long-standing challenge for computational semantics. 
Logic-based models analyse entailment in terms of possible worlds (interpretations, or situations) where a premise P entails a hypothesis H iff in all worlds where P is true, H is also true. Statistical models view this relationship probabilistically, addressing it in terms of whether a human would likely infer H from P. In this paper, we wish to bridge these two perspectives, by arguing for a visually-grounded version of the Textual Entailment task. Specifically, we ask whether models can perform better if, in addition to P and H, there is also an image (corresponding to the relevant "world" or "situation"). We use a multimodal version of the SNLI dataset [2] and we compare "blind" and visually-augmented models of textual entailment. We show that visual information is beneficial, but we also conduct an in-depth error analysis that reveals that current multimodal models are not performing "grounding" in an optimal fashion.

## Running the BiMPM models
### Prerequisites
1. Tensorflow
2. Flick30k + Keras and its [VGG16 pretrained models](https://github.com/fchollet/deep-learning-models) or download [feature file](bimpm/image_features/vgg_feats_77512_2.npy)
3. [Glove embeddings]

### Training
File .config contains all the settings and hyperparameters for training. In order to run:
1. Obtain the image features by either extracting using Keras and Flickr30k dataset or download the features file (git-lfs)
   - To extract image features by yourself, specify location of Flickr30k dataset in [image_utils.py](bimpm/image_utils.py)
        and run ``` python image_utils.py```
2. Specify location of the embedding in the config file
3. ```python main.py --config_file=file_config_name_here.config```

### Evaluation
Trained models are saved in [models directory](bimpm/models). If you want to run decode only, change `decoding_only` to `true`.

[1] Hoa Trong Vu, Claudio Greco, Aliia Erofeeva, Somayeh Jafaritazehjan, Guido Linders, Marc Tanti, Alberto Testoni, Raffaella Bernardi, Albert Gatt - Grounded Textual Entailment
[2] Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)
