from fairseq import options, tasks, checkpoint_utils, progress_bar, utils

import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import roc_auc_score

import logging
import sys


def main():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=sys.stdout,
    )
    parser = options.get_generation_parser(default_task='sequence_tagging')
    args = options.parse_args_and_arch(parser)

    task = tasks.setup_task(args)

    logging.info('Load model from {}'.format(args.path))
    models, _ = checkpoint_utils.load_model_ensemble([args.path], task=task)
    model = models[0]

    if args.fp16:
        model.half()
    model.cuda()

    model.eval()

    task.load_dataset(args.gen_subset)

    test_itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_sentences=16
    ).next_epoch_itr(shuffle=False)

    progress = progress_bar.build_progress_bar(
        args, test_itr)

    all_probs = []
    all_auc = []
    for sample in progress:
        sample = utils.move_to_cuda(sample)

        with torch.no_grad():
            if task.args.roberta_base:
                src_tokens = sample['net_input']['src_tokens']
                src_tokens = utils.convert_padding_direction(src_tokens, task.source_dictionary.pad(), left_to_right=True)
                if src_tokens.size(1) > 768:
                    split1 = src_tokens[:, :512]
                    split2 = src_tokens[:, 256: 768]
                    split3 = src_tokens[:, 512: 1024]

                    split1 = task.base_model.extract_features(split1)
                    split2 = task.base_model.extract_features(split2)
                    split3 = task.base_model.extract_features(split3)

                    encoder_feature = torch.cat([split1, split2[:, 256:, :], split3[:, 256:, :]], dim=1)
                elif src_tokens.size(1) > 512:
                    split1 = src_tokens[:, :512]
                    split2 = src_tokens[:, 256: 768]
                    split1 = task.base_model.extract_features(split1)
                    split2 = task.base_model.extract_features(split2)
                    encoder_feature = torch.cat([split1, split2[:, 256:, :]], dim=1)
                else:
                    encoder_feature = task.base_model.extract_features(src_tokens)
            else:
                src_tokens = sample['net_input']['src_tokens']
                src_tokens = utils.convert_padding_direction(src_tokens, task.source_dictionary.pad(),
                                                             left_to_right=True)
                encoder_feature = task.base_model.extract_features(src_tokens)

            if 'sent_length' in sample:
                sent_input = [encoder_feature[i].split(sample['sent_length'][i]) for i in
                              range(encoder_feature.size(0))]
                sent_input = [torch.cat([split.mean(dim=0, keepdim=True) for split in splits[:-1]], dim=0) for splits in
                              sent_input]

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

            net_output = model(**sample['net_input'])
            probs = net_output.squeeze(dim=-1)
            label_target = model.get_targets(sample, net_output)

            bsz = probs.size(0)
            for i in range(bsz):
                id = sample['id'][i]
                length = sample['net_input']['lengths'][i]

                prob = probs[i, :length]
                target = label_target[i, :length]

                try:
                    auc = roc_auc_score(target.tolist(), prob.tolist())
                    all_auc.append(auc)
                except ValueError:
                    pass
                all_probs.append((id.item(), prob.tolist()[:-1]))

    print('auc {:.5f}'.format(np.mean(all_auc)))

    if args.results_path is not None:
        all_probs = sorted(all_probs, key=lambda x: x[0])
        all_probs = [x[1] for x in all_probs]
        with open(args.results_path, 'w') as f:
            for pred in all_probs:
                f.write(' '.join(['{:.5f}'.format(p) for p in pred]) + '\n')


if __name__ == '__main__':
    main()
