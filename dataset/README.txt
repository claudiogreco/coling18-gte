== Description ==
The V-SNLI dataset [1] has been built by matching each sentence pair of the Stanford Natural Language Inference (SNLI) dataset [2] with the corresponding image coming from the Flickr30K dataset. V-SNLI consists of 565,286 pairs (187,969 neutral, 188,453 contradiction, and 188,864 entailment). Training, test, and development splits have been built according to the splits in SNLI. By construction, V-SNLI contains datapoints such that the premise is always true with respect to the image, whereas the hypothesis can be either true (entailment or neutral cases) or false (contradiction or neutral cases). It has been shown very recently that SNLI contains language bias such that a simple classifier can achieve high accuracy in predicting the three classes just by having as input the hypothesis. The SNLI hard test set, which is a subset of the SNLI test set with ‘hard’ cases, where such a simplistic classifier fails has been released [3]. The V-SNLI hard test set contains the subset of pairs of V-SNLI belonging to the SNLI hard test set.

== Fields ==
gold_label: This is the label taken from SNLI.
sentence1_tokens: The tokens generated for sentence1 according to its parse tree included in SNLI.
sentence2_tokens: The tokens generated for sentence2 according to its parse tree included in SNLI.
image: The image belonging to the Flickr30k dataset of which sentence1 is a caption.
sentence1: The premise caption taken from SNLI.
sentence2: The hypothesis caption taken from SNLI.
pairID: A unique identifier for each (sentence1, sentence2) pair.

== Statistics ==
Total pairs: 565,286 (187,969 neutral, 188,453 contradiction, and 188,864 entailment).
Training pairs: 545,620 (182,167 entailment, 181,938 contradiction, and 181,515 neutral).
Development pairs: 9,842 (3,329 entailment, 3,278 contradiction, and 3,235 neutral).
Test pairs: 9,824 (3,368 entailment, 3,237 contradiction, and 3,219 neutral).
Test hard pairs: 3,262 (1,058 entailment, 1,135 contradiction, and 1,068 neutral).

== References ==
[1] Vu, H., Greco, C., Erofeeva, A., Jafaritazehjan, S., Linders, G., Tanti, M., Testoni, A., Bernardi, R., Gatt, A., 2018. Grounded Textual Entailment. Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018).
[2] Bowman, S.R., Angeli, G., Potts, C. and Manning, C.D., 2015. A large annotated corpus for learning natural language inference. Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP).
[3] Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S.R. and Smith, N.A., 2018. Annotation artifacts in natural language inference data. Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics on Human Language Technology.
