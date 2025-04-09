#!/usr/bin/env bash

# Copyright 2022 University of Texas at Dallas (Szu-Jui Chen)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# options
cleanup=true

log "$0 $*"
if [ $# -ne 0 ] ; then
  log "Error: No positional arguments are required"
  exit 1
fi

. ./db.sh
. utils/parse_options.sh

if [ -z "${FEARLESS_STEPS}" ]; then
    log "Fill the value of 'FEARLESS_STEPS' of db.sh"
    exit 1
fi

audio_dir=${FEARLESS_STEPS}/Audio/Segments/ASR_track2
json_dir=${FEARLESS_STEPS}/Transcripts/ASR_track2
nlsyms=data/nlsyms.txt

for dataset in Train Dev; do
    out=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    log "local/prepare_data.sh --cleanup $cleanup , preparing ${dataset} set"
    local/prepare_data.sh --cleanup $cleanup ${audio_dir}/${dataset} ${json_dir}/${dataset} data/${out}
done

# We then prepare the Eval data
log "local/prepare_audio_only_data.sh --cleanup $cleanup , preparing Eval set"
local/prepare_audio_only_data.sh --cleanup $cleanup ${audio_dir}/Eval data/eval

# remove too short utterence (utt <= 0.15s) in eval set. Total 12 utterences.
sed -i.bak '/03499/d;/03031/d;/06124/d;/01575/d;/18580/d;/10566/d;/01026/d;/00188/d;/21757/d;/03059/d;/11087/d;/13492/d' data/eval/text
utils/fix_data_dir.sh data/eval

# upsample audio from 8k to 16k to make a recipe consistent with others
for x in train dev eval; do
    sed -i.bak -e "s/ / sox -R -t wav /" data/${x}/wav.scp
    sed -i.bak -e "s/$/-t wav - rate 16000 dither | /" data/${x}/wav.scp
done

log "Create non linguistic symbols: ${nlsyms}"
cut -f 2- data/train/text | grep -o -P '\[.*?\]|\<.*?\>' | sort | uniq > ${nlsyms}
cat ${nlsyms}
