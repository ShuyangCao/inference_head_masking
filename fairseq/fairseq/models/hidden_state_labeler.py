import torch
import torch.nn as nn

import torch.nn.functional as F

from fairseq.models import (
    BaseFairseqModel,
    register_model_architecture,
    register_model
)


@register_model('rnn_hidden_state_labeler')
class RNNHiddenStateLabeler(BaseFairseqModel):
    def __init__(self, attribute_name, extractor, projection, dropout):
        super().__init__()
        self.attribute_name = attribute_name
        self.extractor = extractor
        self.projection = projection
        self.dropout = dropout

    @staticmethod
    def add_args(parser):
        parser.add_argument('--attribute-name', metavar='ATTRIBUTE NAME',
                            required=True)
        parser.add_argument('--extractor', metavar='EXTRACTOR',
                            choices=['lstm', 'gru', 'linear'])
        parser.add_argument('--input-dim', type=int, metavar='N')
        parser.add_argument('--hidden-size', type=int, metavar='N')
        parser.add_argument('--inner-dim', type=int, metavar='N')
        parser.add_argument('--dropout', type=float, metavar='D')

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        assert args.attribute_name is not None
        if args.extractor == 'lstm':
            extractor = nn.LSTM(args.input_dim, args.hidden_size, bidirectional=True)
            proj_dim = args.hidden_size * 2
        elif args.extractor == 'gru':
            extractor = nn.GRU(args.input_dim, args.hidden_size, bidirectional=True)
            proj_dim = args.hidden_size * 2
        elif args.extractor == 'linear':
            extractor = None
            proj_dim = args.input_dim
        else:
            raise NotImplementedError

        projection = nn.Sequential(
            nn.Linear(proj_dim, args.inner_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.inner_dim, 1),
            nn.Sigmoid()
        )

        return cls(args.attribute_name, extractor, projection, args.dropout)

    def forward(self, features, lengths):
        bsz, seqlen, _ = features.size()
        
        x = F.dropout(features, p=self.dropout, training=self.training)

        if self.extractor is not None:
            x = x.transpose(0, 1)

            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)

            packed_out, _ = self.extractor(packed_x)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_out)
            x = x.transpose(0, 1)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        probs = self.projection(x)

        return probs
    
    def get_targets(self, sample, net_output):
        return sample['label']


@register_model_architecture('rnn_hidden_state_labeler', 'rnn_hidden_state_labeler')
def base_architecture(args):
    args.extractor = getattr(args, 'extractor', args.extractor)
    args.input_dim = getattr(args, 'input_dim', 1024)
    args.hidden_size = getattr(args, 'hidden_size', 1024)
    args.inner_dim = getattr(args, 'inner_dim', 512)
    args.dropout = getattr(args, 'dropout', args.dropout)
