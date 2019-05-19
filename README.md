# Grounded Textual Entailment
Dataset and code associated to the paper [Grounded Textual Entailment](https://arxiv.org/pdf/1806.05645.pdf) [1]. A BibTeX entry for the paper is the following:

```
@InProceedings{vu2018grounded,
  title={Grounded Textual Entailment},
  author={Vu, Hoa Trong and Greco, Claudio and Erofeeva, Aliia and Jafaritazehjan, Somayeh and Linders, Guido and Tanti, Marc and Testoni, Alberto and Bernardi, Raffaella and Gatt, Albert},
  booktitle={Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018)},
  year={2018}
}
```

## Abstract
Capturing semantic relations between sentences, such as entailment, is a long-standing challenge for computational semantics. 
Logic-based models analyse entailment in terms of possible worlds (interpretations, or situations) where a premise P entails a hypothesis H iff in all worlds where P is true, H is also true. Statistical models view this relationship probabilistically, addressing it in terms of whether a human would likely infer H from P. In this paper, we wish to bridge these two perspectives, by arguing for a visually-grounded version of the Textual Entailment task. Specifically, we ask whether models can perform better if, in addition to P and H, there is also an image (corresponding to the relevant "world" or "situation"). We use a multimodal version of the SNLI dataset [2] and we compare "blind" and visually-augmented models of textual entailment. We show that visual information is beneficial, but we also conduct an in-depth error analysis that reveals that current multimodal models are not performing "grounding" in an optimal fashion.

## Dataset
The dataset is available [here](https://drive.google.com/file/d/1v5HZtSFF0FH-5mr5sHHjabI51lDJFZjt/view?usp=sharing).

## Pre-trained models
The pre-trained models are available [here](https://drive.google.com/drive/folders/1CoDmxA0XPN_ddKPs2KDSkfMbEcgGgkHI?usp=sharing)

## Running the BiMPM models
### Prerequisites
1. Tensorflow
2. Flick30k + Keras and its [VGG16 pretrained models](https://github.com/fchollet/deep-learning-models) or download [feature files](https://drive.google.com/file/d/1_PteTR8vHF8kC9x1LYnW0b1q9A2ggAz3/view?usp=sharing)
3. [Glove embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip)

### Training
File .config contains all the settings and hyperparameters for training. In order to run:
1. Obtain the image features by either extracting using Keras and Flickr30k dataset or download the features file (git-lfs)
   - To extract image features by yourself, specify location of Flickr30k dataset in [image_utils.py](bimpm/image_utils.py)
        and run ``` python image_utils.py```
2. Specify location of the embedding in the config file
3. ```python main.py --config_file=file_config_name_here.config```

### Evaluation
Trained models are saved in [models directory](bimpm/models). If you want to run decode only, change `decoding_only` to `true`.

## Bibliography
[1] Hoa Trong Vu, Claudio Greco, Aliia Erofeeva, Somayeh Jafaritazehjan, Guido Linders, Marc Tanti, Alberto Testoni, Raffaella Bernardi, Albert Gatt. 2018. Grounded Textual Entailment. In Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018).

[2] Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP).
