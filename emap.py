'''
Example implementation of EMAP
'''
import numpy as np
import collections


def emap(idx2logits):
    '''Example implementation of EMAP (more efficient ones exist)

    inputs:
      idx2logits: This nested dictionary maps from image/text indices
        function evals, i.e., idx2logits[i][j] = f(t_i, v_j)

    returns:
      projected_preds: a numpy array where projected_preds[i]
        corresponds to \hat f(t_i, v_i).
    '''
    all_logits = []
    for k, v in idx2logits.items():
        all_logits.extend(v.values())
    all_logits = np.vstack(all_logits)
    logits_mean = np.mean(all_logits, axis=0)

    reversed_idx2logits = collections.defaultdict(dict)
    for i in range(len(idx2logits)):
        for j in range(len(idx2logits[i])):
            reversed_idx2logits[j][i] = idx2logits[i][j]

    projected_preds = []
    for idx in range(len(idx2logits)):
        pred = np.mean(np.vstack(list(idx2logits[idx].values())), axis=0)
        pred += np.mean(np.vstack(list(reversed_idx2logits[idx].values())), axis=0)
        pred -= logits_mean
        projected_preds.append(pred)

    projected_preds = np.vstack(projected_preds)
    return projected_preds


def test_from_paper():
    '''tests the emap code using the worked example in the appendix'''

    idx2logits = collections.defaultdict(dict)

    idx2logits[0][0] = -1.3 ; idx2logits[0][1] = .3; idx2logits[0][2] = -.2;
    idx2logits[1][0] = .8; idx2logits[1][1] = 3.0; idx2logits[1][2] = 1.1;
    idx2logits[2][0] = 1.1; idx2logits[2][1] = -.1; idx2logits[2][2] = .7;

    print('original model predictions:')
    print([idx2logits[idx][idx] for idx in range(3)])
    print('emap:')
    print(emap(idx2logits))


if __name__ == '__main__':
    test_from_paper()
