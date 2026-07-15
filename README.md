# AfriNews Multilingual Classification

**What:** Predicts one of seven news topics for each article from the 384 prepared
`tx_*` features (L2-normalized, noise-perturbed hashed character n-grams).

**Run:** `python train.py` from the task workspace root. Reads `public/{train,test,train_unlabeled}.csv`
and writes `submission/submission.csv`. CPU only, a few minutes.

**Approach:** The features are unit-norm n-gram vectors, so cosine/RBF neighbor models beat
linear and tree models by a wide margin (CV ~0.60 vs ~0.47). A logistic meta-learner stacks
three views: an RBF-SVM (`C=10, gamma=4`), a cosine kNN (`k=11`), and transductive label
spreading over a kNN graph built from the labeled rows plus the unlabeled pool and the test
rows. Because the features are unit-norm, a Euclidean kNN graph equals a cosine graph, so
label spreading follows the same geometry as the kNN model but lets topic labels flow through
the unlabeled/test manifold; it is the largest single contributor to the worst-language F1,
which the metric weights heavily. The meta-learner also takes a language one-hot so it can
shift toward each language's class prior. Each language uses only a fixed subset of the seven
topics (e.g. Amharic is never 1/4/6), so probabilities are masked to a language's observed
topics before the argmax. Base probabilities for stacking are produced out-of-fold (5-fold,
stratified by language and topic), with the label-spreading pass treating the held-out fold as
unlabeled to mirror how the test rows are scored. Hard self-training was tested and discarded
(pseudo-label noise lowered CV); soft label spreading used only as a meta feature helps.

**Score:** Language-balanced weighted F1 = **0.683** on the prepared test set with the shipped
seed; mean over five seeds is 0.684 (sd 0.004, min 0.679). Ablation on the same test set:
SVM+kNN average 0.639 -> add language-aware stacking 0.653 -> add transductive label spreading
0.683, which also raises the worst language (`fra`) from 0.53 to 0.60.
