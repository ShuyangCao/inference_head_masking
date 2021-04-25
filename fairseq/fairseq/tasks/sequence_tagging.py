from argparse import Namespace
import logging
import os

import torch
import torch.nn as nn

from fairseq import options, tasks, utils
from fairseq.data import (
    indexed_dataset,
    data_utils,
    iterators,
    SourceLabelDataset,
    AppendTokenDataset,
    TruncateDataset,
    StripTokenDataset,
    PrependTokenDataset
)

from fairseq.tasks import FairseqTask, register_task

logger = logging.getLogger(__name__)


def load_source_label_dataset(
        data_path, split,
        src, src_dict, tgt,
        label_suffix, sentence_level,
        dataset_impl,
        left_pad_source, max_source_positions,
        prepend_bos=False,
        truncate_source=False,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    # infer langcode
    if split_exists(split, src, tgt, src, data_path):
        prefix = os.path.join(data_path, '{}.{}-{}.'.format(split, src, tgt))
    elif split_exists(split, tgt, src, src, data_path):
        prefix = os.path.join(data_path, '{}.{}-{}.'.format(split, tgt, src))
    else:
        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

    src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
    if truncate_source:
        src_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDataset(src_dataset, src_dict.eos()),
                max_source_positions - 1,
            ),
            src_dict.eos(),
        )

    sent_lengths = None
    sent_labels = None
    token_labels = None
    if sentence_level:
        with open(prefix + 'sentlengths') as f:
            sent_lengths = f.readlines()
        sent_lengths = [[int(l) for l in sent_length.strip().split(' ') if l] for sent_length in sent_lengths]

        with open(prefix + label_suffix) as f:
            selected_sid = f.readlines()
        selected_sid = [[int(sid) for sid in sids.strip().split(' ') if sid] for sids in selected_sid]

        def process_sentence(lengths, selected_sents):
            new_lengths = []
            new_sent_labels = []
            for sent_length, selected_sent in zip(lengths, selected_sents):
                new_length = []
                new_sent_label = []
                for i, length in enumerate(sent_length):
                    if i in selected_sent:
                        new_sent_label.append(1)
                    else:
                        new_sent_label.append(0)
                    accumulated_length = sum(new_length + [length])
                    exceed_length = accumulated_length - (max_source_positions - 1)
                    if exceed_length >= 0:
                        new_length.append(length - exceed_length)
                        break
                    else:
                        new_length.append(length)
                new_length.append(1)
                new_lengths.append(new_length)
                new_sent_labels.append(torch.FloatTensor(new_sent_label))
            return new_lengths, new_sent_labels

        sent_lengths, sent_labels = process_sentence(sent_lengths, selected_sid)
    else:
        with open(prefix + label_suffix) as f:
            token_labels = f.readlines()
        token_labels = [torch.FloatTensor([int(label) for label in token_label.strip().split(' ')[:max_source_positions - 1] if label] + [0]) for token_label in token_labels]
        
    logger.info('{} {} {}-{} {} examples'.format(
        data_path, split, src, tgt, len(src_dataset)
    ))

    if prepend_bos:
        assert hasattr(src_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())

    return SourceLabelDataset(
        src_dataset, src_dataset.sizes, src_dict,
        token_labels, sent_labels, sent_lengths,
        left_pad_source=left_pad_source,
        max_source_positions=max_source_positions,
    )


@register_task('sequence_tagging')
class SequenceTaggingTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument('data')
        parser.add_argument('--source-lang', default='src')
        parser.add_argument('--target-lang', default='tgt')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--max-input-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the input sequence')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate input sequence to max-input-positions')

        # labeler options
        parser.add_argument('--sentence-level', action='store_true',
                            help='sentence level labeler')
        parser.add_argument('--label', type=str, metavar='LABEL SUFFIX',
                            required=True, help='suffix of the label file')

        parser.add_argument('--base-model')
        parser.add_argument('--roberta-base', action='store_true')

    def __init__(self, args, src_dict, base_model):
        super().__init__(args)
        self.src_dict = src_dict
        self.base_model = base_model

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)

        src_dict = cls.load_dictionary(os.path.join(args.data, 'dict.{}.txt'.format(args.source_lang)))

        logger.info('{} level labeler'.format('sentence' if args.sentence_level else 'token'))
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))

        assert args.base_model is not None
        assert os.path.exists(args.base_model)

        use_cuda = torch.cuda.is_available() and not args.cpu

        from fairseq.checkpoint_utils import load_checkpoint_to_cpu
        if args.roberta_base:
            from fairseq.models.roberta import RobertaModel
            base_model = RobertaModel.from_pretrained(args.base_model)
        elif os.path.isdir(args.base_model):
            from fairseq.models.bart import BARTModel
            base_model = BARTModel.from_pretrained(args.base_model)
        else:
            base_model_state = load_checkpoint_to_cpu(args.base_model)
            base_model_args = base_model_state['args']

            base_model_task = tasks.setup_task(base_model_args)
            base_model = base_model_task.build_model(base_model_args)
            base_model.load_state_dict(base_model_state['model'], strict=True)
            base_model.make_generation_fast_()
        if args.fp16:
            base_model.half()
        if use_cuda:
            base_model.cuda()

        base_model.eval()

        return cls(args, src_dict, base_model)

    def load_dataset(self, split, combine=False, **kwargs):
        src = self.args.source_lang
        tgt = self.args.target_lang

        logger.info('label type: {}'.format(self.args.label))

        self.datasets[split] = load_source_label_dataset(
            self.args.data, split,
            src, self.src_dict, tgt,
            self.args.label, self.args.sentence_level,
            self.args.dataset_impl,
            left_pad_source=self.args.left_pad_source,
            max_source_positions=self.args.max_input_positions,
            truncate_source=self.args.truncate_source,
        )

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        # get the features from base model
        # NOTE: the input is left pad, need to change to right pad before feeding into the labeling model
        with torch.no_grad():
            if self.args.roberta_base:
                src_tokens = sample['net_input']['src_tokens']
                src_tokens = utils.convert_padding_direction(src_tokens, self.source_dictionary.pad(), left_to_right=True)
                if src_tokens.size(1) > 768:
                    split1 = src_tokens[:, :512]
                    split2 = src_tokens[:, 256: 768]
                    split3 = src_tokens[:, 512: 1024]

                    split1 = self.base_model.extract_features(split1)
                    split2 = self.base_model.extract_features(split2)
                    split3 = self.base_model.extract_features(split3)

                    encoder_feature = torch.cat([split1, split2[:, 256:, :], split3[:, 256:, :]], dim=1)
                elif src_tokens.size(1) > 512:
                    split1 = src_tokens[:, :512]
                    split2 = src_tokens[:, 256: 768]
                    split1 = self.base_model.extract_features(split1)
                    split2 = self.base_model.extract_features(split2)
                    encoder_feature = torch.cat([split1, split2[:, 256:, :]], dim=1)
                else:
                    encoder_feature = self.base_model.extract_features(src_tokens)
            else:
                src_tokens = sample['net_input']['src_tokens']
                src_tokens = utils.convert_padding_direction(src_tokens, self.source_dictionary.pad(),
                                                             left_to_right=True)
                encoder_feature = self.base_model.extract_features(src_tokens)

            if 'sent_length' in sample:
                sent_input = [encoder_feature[i].split(sample['sent_length'][i]) for i in range(encoder_feature.size(0))]
                sent_input = [torch.cat([split.mean(dim=0, keepdim=True) for split in splits[:-1]], dim=0)
                              if len(splits) > 1 else splits[0].mean(dim=0, keepdim=True) for splits in sent_input]

                # now the sent_input are of different length
                sent_input = nn.utils.rnn.pad_sequence(sent_input, batch_first=True)
                lengths = [len(sent_length) - 1 for sent_length in sample['sent_length']]
                
                sample['net_input'] = {
                    'features': sent_input,
                    'lengths': lengths
                }
            else:
                sample['net_input'] = {
                    'features': encoder_feature,
                    'lengths': sample['net_input']['src_lengths'].tolist()
                }

        # forward to discriminator
        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            if self.args.roberta_base:
                src_tokens = sample['net_input']['src_tokens']
                src_tokens = utils.convert_padding_direction(src_tokens, self.source_dictionary.pad(), left_to_right=True)
                if src_tokens.size(1) > 768:
                    split1 = src_tokens[:, :512]
                    split2 = src_tokens[:, 256: 768]
                    split3 = src_tokens[:, 512: 1024]

                    split1 = self.base_model.extract_features(split1)
                    split2 = self.base_model.extract_features(split2)
                    split3 = self.base_model.extract_features(split3)

                    encoder_feature = torch.cat([split1, split2[:, 256:, :], split3[:, 256:, :]], dim=1)
                elif src_tokens.size(1) > 512:
                    split1 = src_tokens[:, :512]
                    split2 = src_tokens[:, 256: 768]
                    split1 = self.base_model.extract_features(split1)
                    split2 = self.base_model.extract_features(split2)
                    encoder_feature = torch.cat([split1, split2[:, 256:, :]], dim=1)
                else:
                    encoder_feature = self.base_model.extract_features(src_tokens)
            else:
                src_tokens = sample['net_input']['src_tokens']
                src_tokens = utils.convert_padding_direction(src_tokens, self.source_dictionary.pad(),
                                                             left_to_right=True)
                encoder_feature = self.base_model.extract_features(src_tokens)
                # encoder_out = self.base_model.encoder(**sample['net_input'])
                # encoder_feature = encoder_out.encoder_out
                #
                # encoder_feature = encoder_feature.transpose(0, 1)

            if 'sent_length' in sample:
                sent_input = [encoder_feature[i].split(sample['sent_length'][i]) for i in
                              range(encoder_feature.size(0))]
                sent_input = [torch.cat([split.mean(dim=0, keepdim=True) for split in splits[:-1]], dim=0)
                              if len(splits) > 1 else splits[0].mean(dim=0, keepdim=True) for splits in sent_input]

                # now the sent_input are of different length
                sent_input = nn.utils.rnn.pad_sequence(sent_input, batch_first=True)
                lengths = [len(sent_length) - 1 for sent_length in sample['sent_length']]

                sample['net_input'] = {
                    'features': sent_input,
                    'lengths': lengths
                }
            else:
                sample['net_input'] = {
                    'features': encoder_feature,
                    'lengths': sample['net_input']['src_lengths'].tolist()
                }
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def build_model(self, args):
        assert args.arch == 'rnn_hidden_state_labeler'
        return super().build_model(args)

    def max_positions(self):
        if self.args.roberta_base:
            return 1024
        try:
            return self.base_model.encoder.max_positions()
        except:
            return self.base_model.max_positions

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return None

