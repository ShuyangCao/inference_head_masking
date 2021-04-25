# Attention Head Masking for Inference Time Content Selectionin Abstractive Summarization

Code for the paper [Attention Head Masking for Inference Time Content Selectionin Abstractive Summarization](https://arxiv.org/abs/2104.02205)

## Requirements

Our code uses PyTorch version `1.4`. Higher versions might also work, but we haven't tested them.

After having PyTorch installed, please install our modified Fairseq library.

```shell
cd fairseq
pip install -e .
```

To run the sequence tagger, please also download the `roberta.base` model from the [offical Fairseq repository](https://github.com/pytorch/fairseq/tree/master/examples/roberta). 

## Decode with Head Masking

Our sequence taggers and fine-tuned summarization models can be downloaded from [here](https://drive.google.com/file/d/1JEcM5WUMO-w5eQKo215Le5iRqY-faVuo/view?usp=sharing).
Binarized datasets and the tagging results produced by our content selectors are also provided.

#### Decode with Selection Labels

To decode with selection labels, please make sure `test.source-target.sl.token.roberta.full.nobpe` is in the binarized dataset directory.

```shell
cd scripts
chmod +x test_head_masking_cnndm.sh
./test_head_masking_cnndm.sh \
    /path/to/binarized_cnndm \
    /path/to/cnndm_bart/checkpoint_best.pt \
    /path/to/savedir
```

After decoding, get the text output from the BPE output by:

```shell
cd data_processing
python convert_output.py --generate-dir /path/to/savedir
```

The text output will be saved in `/path/to/savedir/formatted-test.txt`.

To apply masks at different layers or heads, 
change the `SELECT_HEADS` and `SELECT_LAYER` variables in `scripts/test_head_masking_cnndm.sh`.

#### Create Selection Labels

Create selection labels with our sequence tagger, 
please make sure the oracle selection label `test.source-target.fragment` is in the binarized dataset directory.

```shell
cd scripts
python run_sequence_label.py \
    /path/to/binarized_cnndm \ 
    --path /path/to/cnndm_tagger/checkpoint_best.pt \
    --base-model /path/to/roberta.base \
    --roberta-base --source-lang source --target-lang target \
    --label fragmentnobpe --truncate-source \
    --results-path /path/to/binarized_cnndm/test.source-target.sl.token.roberta.full
```

Ensure that words with multiple BPE units have the same selection score:

```shell
cd data_processing
python convert_nobpe_label.py \
  --src /path/to/binarized_cnndm/test.source \
  --src-bpe /path/to/binarized_cnndm/test.bpe.source \
  --label /path/to/binarized_cnndm/test.source-target.sl.token.roberta.full \
  --out /path/to/binarized_cnndm/test.source-target.sl.token.roberta.full.nobpe
```

## Evaluation

To evaluate the generated summaries, we use `files2rouge` ([link](https://github.com/pltrdy/files2rouge)).

```shell
export CLASSPATH=/path/to/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar:$CLASSPATH

cat /path/to/savedir/formatted-test.txt | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > /path/to/savedir/tokenized-test.txt
cat /path/to/binarized_cnndm/test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > /path/to/binarized_cnndm/tokenized.test.target
files2rouge /path/to/binarized_cnndm/tokenized.test.target /path/to/savedir/tokenized-test.txt
```
