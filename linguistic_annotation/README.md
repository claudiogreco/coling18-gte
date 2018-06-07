# Linguistic annotation of V-SNLI

This dataset contains linguistic annotation of the V-SNLI test set \[1].
Following the error analysis approach described in recent work \[2,3],
a new list of linguistic features has been compiled, that can be of interest
when evaluating RTE models. Some of these are assigned manually (`VSNLI_1.0_manual_tags.tsv`),
while others are detected automatically (`VSNLI_1.0_auto_tags.tsv`).

### Format

* `pairID`: A unique SNLI identifier for each (hypothesis, premise) pair.
* `tags`: List of tags separated by comma.

### Statistics

* Manual tags: 530 examples (188 entailment, 171 neutral, 171 contradiction).
* Automatic tags: 9,778 examples (3352 entailment, 3218 neutral, 3208 contradiction).

## Manual tags

Tag | Freq | Description
--- | ----: | -----------
Insertion | 167 |Hypothesis (H) contains details and facts not present in the premise (P) (e.g., subjective or emotional judgments).
Generalisation | 166 | One-way entailment, i.e., H entails P.
Entity | 107 |P and H describe different entities (e.g., subject, object, location) or incompatible properties of entities (e.g., color).
Verb | 101 | The sentences describe different, incompatible actions.
World_knowledge | 93 | Commonsense assumptions are needed to understand the relation between sentences (e.g., if there are named entities).
Quantifier | 91 | The sentences contain numbers or quantifiers (e.g., *all, no, some, both, group*).
Image_mismatch | 57 | Gold label is incorrect when the image is factored in.
Incorrect | 45 | Incorrect gold label (annotation artifact).
Voice | 3 | The premise is an active/passive transformation of the hypothesis.
Paraphrase | 7 | Two-way entailment, i.e., H entails P and vice versa.
Unrelated | 6 | The sentences are completely unrelated.
Swap | 1 | The sentences' subject and object are swapped from P to H.

## Automatic tags

Automatic tags are identified using WordNet, StanfordNLP parses, or *keywords* search.

Tag | Freq | Description
--- | ----: | -----------
DIFF_TENSE | 7431 | Verbs in hypothesis (H) and premise (P) are in different tenses.
QUANTIFIER | 3779 | H or P contain quantifiers (cardinal numbers, *almost, any, enough, few, group, least, less, many, more, most, much, nearly, several, some*).
PRONOUN | 3203 | H or P contain personal or possessive pronouns.
SYNONYM | 1798 | H or P contain synonyms.
ANTONYM | 882 | H or P contain antonyms.
SUPERLATIVE | 304 | H or P contain superlative forms.
LONG | 303 | H contains >16 tokens or P contains more > 30 tokens.
BARE_NP | 281 | H or P are bare noun phrases.
NEGATION | 185 | H or P contain negation (*n't, neither, never, no, nobody, none, nor, not, nothing, nowhere*).

## References

\[1] Vu, H., Greco, C., Erofeeva, A., Jafaritazehjan, S., Linders, G., Tanti, M., Testoni, A., Bernardi, R., Gatt, A., 2018. Grounded Textual Entailment. In Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018).

\[2] Nikita Nangia, Adina Williams, Angeliki Lazaridou, and Samuel R Bowman.  2017.  The RepEval 2017 Shared Task: Multi-Genre Natural Language Inference with Sentence Representations. arXiv, 1707.08172.

\[3] Adina Williams, Nikita Nangia, and Samuel R. Bowman. 2018. A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. In Proceedings of NAACL.
