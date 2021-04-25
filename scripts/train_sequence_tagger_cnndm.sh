TOTAL_NUM_UPDATES=12000
WARMUP_UPDATES=500
LR=5e-4
MAX_SENTENCES=128
UPDATE_FREQ=1

DATA_PATH=$1
BASE_MODEL=$2
SAVE_PATH=$3

fairseq-train $DATA_PATH \
    --base-model $BASE_MODEL --roberta-base \
    --max-sentences $MAX_SENTENCES \
    --task sequence_tagging \
    --source-lang source --target-lang target \
    --label fragmentnobpe \
    --truncate-source \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch rnn_hidden_state_labeler \
    --extractor linear --dropout 0.1 --input-dim 768 \
    --attribute-name cs \
    --criterion binary_sequence_labeling_loss \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 --fp16 \
    --lr-scheduler polynomial_decay --lr $LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --validate-interval 1 \
    --num-workers 0 \
    --save-dir $SAVE_PATH \
    --no-epoch-checkpoints --no-last-checkpoints \
    --no-save-optimizer-state;