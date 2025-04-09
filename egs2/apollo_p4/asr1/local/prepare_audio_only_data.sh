#!/usr/bin/env bash

# Copyright 2020 University of Texas at Dallas (Szu-Jui Chen)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# options
cleanup=true

. ./utils/parse_options.sh
. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 2 ] ; then
  echo >&2 "Error: unexpected number of arguments"
  echo -e >&2 "Usage:\n $0 [options] <audio-dir> <output-dir>"
  exit 1
fi

set -e -o pipefail

audio_dir=$1
dir=$2

echo "$0: Creating data directory $dir"
mkdir -p $dir

# 'spkID'-'file name' is the uttID 
# e.g., NETWORKS-FS02_ASR_track2_train_00001 /corpus/A11_100hr/FS02_Challenge_Data/ \
# Audio/Segments/ASR_track2/Train/FS02_ASR_track2_train_00001.wav
find -L $audio_dir -name "*.wav" | \
  perl -ne '{ 
    chomp;
    $path=$_;
    next unless $path;
    @F = split"/", $path;
    @F2 = split"_", $path;
    ($f = $F[@F-1]) =~ s/.wav//, ($p = $F2[@F2-1]) =~ s/.wav//;
    print "$p-$f $path \n"
}' | sort > $dir/wav.scp

# prepare 'utt2spk' and 'spk2utt'
cut -d'-' -f1 $dir/wav.scp > $dir/spk_list.tmp
cut -d' ' -f1 $dir/wav.scp > $dir/utt_id.tmp
paste -d ' ' $dir/utt_id.tmp $dir/spk_list.tmp | sort > $dir/utt2spk

utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

# generate empty text
cut -d' ' -f1 $dir/wav.scp > $dir/text.tmp
sed -e 's/$/ a/' $dir/text.tmp | sort > $dir/text

$cleanup && rm -f $dir/*.tmp

# Check that data dirs are okay
utils/validate_data_dir.sh --no-feats $dir || exit 1

echo "$0 Finished preparing $dir"

