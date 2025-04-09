#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev eval"
bpe_nlsyms="[vocalization],[unk]"

asr_config=conf/tuning/train_asr_conformer_s3prlfrontend_hubert_lr0.002_warm1w8.yaml
lm_config=conf/tuning/lm/train_lm_transformer11.yaml
inference_config=conf/decode.yaml

./asr.sh \
    --lang en \
    --ngpu 8 \
    --nbpe 500 \
    --nlsyms_txt data/nlsyms.txt \
    --feats_normalize uttmvn \
    --bpe_nlsyms "${bpe_nlsyms}" \
    --max_wav_duration 30 \
    --min_wav_duration 0.15 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format wav \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "dump/raw/train_sp/text" "$@"
