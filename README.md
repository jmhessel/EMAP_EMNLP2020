# What's in here?

This repo contains (a very short amount of) code from: [Does my multimodal model learn cross-modal interactions? It's harder to tell than you might think!](https://arxiv.org/abs/2010.06572) The bibtex is:

```
@inproceedings{hessel2020cats,
	title={Does my multimodal model learn cross-modal interactions? It's harder to tell than you might think!},
	author={Hessel, Jack and Lee, Lillian},
	booktitle={EMNLP},
	year={2020}
}
```

## what does this code do?

The code here implements a minimal version of **E**mpirical
**M**ultimodally-**A**dditive **P**rojections (EMAP), as described in our
EMNLP paper.

TL;DR: if you have a multimodal classification task, EMAP can provide
insight in whether or not your model is using conditional, cross-modal
interactions (like cross-modal attention) to make more accurate
predictions (or not). For many multimodal tasks, we imagine that our
algorithms are doing the types of inferences that we do, carefully
comparing different aspects of images and text and then making an
informed decision based on that interaction. *But are they?* EMAP can
tell you! (see the paper for more details!)

## how do you use it?

Say you had `N` evaluation datapoints consisting of text/visual
pairings `ti, vi`. The `emap` function in `emap.py` assumes as input a
dictionary that maps from indices `i,j \in {1...N}` to the output
logits of your predictor `f` evaluated on `ti,vj` for each class. The
function returns the projected predictions of your predictor's EMAP,
which are as-close-as-possible to your original model's predictions,
but only have additive structure.