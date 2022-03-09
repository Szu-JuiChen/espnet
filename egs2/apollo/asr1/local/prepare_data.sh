#!/usr/bin/env bash

# Copyright 2020 University of Texas at Dallas (Szu-Jui Chen)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# options
cleanup=true

. ./utils/parse_options.sh
. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 3 ] ; then
  echo >&2 "Error: unexpected number of arguments"
  echo -e >&2 "Usage:\n $0 [options] <audio-dir> <json-dir> <output-dir>"
  exit 1
fi

set -e -o pipefail

audio_dir=$1
json_dir=$2
dir=$3

echo "$0: Creating data directory $dir"
mkdir -p $dir

# make the spkID a prefix of uttID
cut -d " " -f 1 $json_dir/*transcription* > $dir/text1.tmp #this is the utt id list
cut -d " " -f 2- $json_dir/*transcription* | sed -e 's/\[unk\]/<unk>/g' | tr '[:upper:]' '[:lower:]' \
    > $dir/text2.tmp
paste -d ' ' $dir/text1.tmp $dir/text2.tmp > $dir/text.tmp
cut -d " " -f 2 $json_dir/*uttID* > $dir/spk_list.tmp
paste -d '-' $dir/spk_list.tmp $dir/text.tmp | sort > $dir/text

# 'spkID'-'file name' is the uttID 
# e.g., NETWORKS-FS02_ASR_track2_train_00001 /corpus/A11_100hr/FS02_Challenge_Data/ \
# Audio/Segments/ASR_track2/Train/FS02_ASR_track2_train_00001.wav
find -L $audio_dir -name "*.wav" | \
  perl -ne '{
    chomp;
    $path = $_;
    next unless $path;
    @F = split"/", $path;
    ($f = $F[@F-1]) =~ s/.wav//;
    print "$f $path \n"
  }' | sort > $dir/wav.scp.tmp

paste -d '-' $dir/spk_list.tmp $dir/wav.scp.tmp | sort > $dir/wav.scp

# prepare 'utt2spk' and 'spk2utt'
paste -d '-' $dir/spk_list.tmp $json_dir/*uttID* | sort > $dir/utt2spk

utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

$cleanup && rm -f $dir/*.tmp

# Check that data dirs are okay
utils/validate_data_dir.sh --no-feats $dir || exit 1

echo "$0 Finished preparing $dir"

