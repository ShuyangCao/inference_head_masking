import argparse
import os
import regex as re
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from collections import Counter, OrderedDict, defaultdict

bpe_encoder = GPT2BPE(None).bpe

tokenizer = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src')
    parser.add_argument('--src-bpe')
    parser.add_argument('--label')
    parser.add_argument('--keep-first', action='store_true')
    parser.add_argument('--out')
    args = parser.parse_args()

    with open(args.src) as f:
        srcs = f.readlines()

    with open(args.src_bpe) as f:
        src_bpes = f.readlines()

    with open(args.label) as f:
        labels = f.readlines()

    with open(args.out, 'w') as f:
        for src, src_bpe, label in zip(srcs, src_bpes, labels):
            s_gpt_token = re.findall(tokenizer, src.strip())
            s_token = [token.strip() for token in s_gpt_token]
            s_num = []
            for token in s_gpt_token:
                x = ''.join(bpe_encoder.byte_encoder[b] for b in token.encode('utf-8'))
                num = len(bpe_encoder.bpe(x).split(' '))
                s_num.append(num)
            label = label.strip().split(' ')

            token_idx = 0
            bpe_end_idx = s_num[token_idx]

            new_label = []
            new_label_candidates = []

            token_prob = defaultdict(float)

            highest_prob = 0
            bpe_len = 0
            for i, p in enumerate(label):
                if i >= bpe_end_idx:
                    new_label_candidates.append((highest_prob, bpe_len, s_token[token_idx]))
                    if highest_prob > token_prob[s_token[token_idx]]:
                        token_prob[s_token[token_idx]] = highest_prob

                    token_idx += 1
                    bpe_end_idx += s_num[token_idx]

                    highest_prob = 0
                    bpe_len = 0

                p = float(p)
                if p > highest_prob:
                    highest_prob = p

                bpe_len += 1

            if bpe_len > 0:
                new_label_candidates.append((highest_prob, bpe_len, s_token[token_idx]))
                if highest_prob > token_prob[s_token[token_idx]]:
                    token_prob[s_token[token_idx]] = highest_prob

            include_token = []
            for candidate in new_label_candidates:
                if args.keep_first:
                    if candidate[2] in include_token:
                        new_label.extend([0.] * candidate[1])
                        continue
                    if candidate[1] >= token_prob[candidate[2]]:
                        new_label.extend([candidate[0]] * candidate[1])
                        include_token.append(candidate[2])
                    else:
                        new_label.extend([0.] * candidate[1])
                else:
                    new_label.extend([candidate[0]] * candidate[1])
                    include_token.append(candidate[2])

            f.write(' '.join(['{:.4f}'.format(l) for l in new_label]))

            f.write('\n')


if __name__ == '__main__':
    main()