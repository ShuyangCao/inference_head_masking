BEAM_SIZE=6
MAX_LEN_B=60
MIN_LEN=10
LEN_PEN=1.0
MAX_SENTENCES=32

SOURCE_LABEL="sl.token.roberta.full.nobpe"
SELECT_HEADS="3,11,13,14,2,4,8,15,9,6,1,7"
SELECT_LAYER=3
THRESHOLD=0.18

DATA_PATH=$1
BASE_MODEL=$2
RESULT_PATH=$3

fairseq-generate $DATA_PATH \
    --results-path $RESULT_PATH \
    --source-lang source --target-lang target \
    --task inference_head_masking \
    --path $BASE_MODEL \
    --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN \
    --no-repeat-ngram-size 3 \
    --max-sentences $MAX_SENTENCES \
    --required-batch-size-multiple 1 \
    --truncate-source --gen-subset test \
    --source-label $SOURCE_LABEL --label-threshold $THRESHOLD \
    --mask-heads $SELECT_HEADS --mask-layer $SELECT_LAYER;
