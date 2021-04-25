# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

from fairseq import utils, metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('binary_sequence_labeling_loss')
class BinarySequenceLabelingLossCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        probs = net_output.squeeze(dim=-1)
        target = model.get_targets(sample, net_output)

        # assert ((probs >= 0) & (probs <= 1)).all(), sample['net_input']['features']

        loss = F.binary_cross_entropy(probs.float(), target.float(), reduction='none')

        ntokens = 0
        auc = 0.
        nauc = 0
        for i, length in enumerate(sample['net_input']['lengths']):
            ntokens += length
            loss[i, length:] = 0.
            try:
                auc += roc_auc_score(target[i, :length].tolist(), probs[i, :length].tolist())
                nauc += 1
            except ValueError:
                pass

        if reduce:
            loss = loss.sum()

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': ntokens,
            'nsentences': probs.size(0),
            'sample_size': ntokens,
            'auc': auc,
            'nauc': nauc
        }
        return loss, ntokens, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        auc = sum(log.get('auc', 0) for log in logging_outputs)
        nauc = sum(log.get('nauc', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('auc', (auc / nauc) if nauc != 0 else 0)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: round(2 ** meters['nll_loss'].avg, 3))
        else:
            metrics.log_derived('ppl', lambda meters: round(2 ** meters['loss'].avg, 3))
