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
  echo -e >&2 "Usage:\n $0 [options] <audio-dir> <segment_dir> <output-dir>"
  exit 1
fi

set -e -o pipefail

audio_dir=$1
segment_dir=$2
dir=$3

echo "$0: Creating data directory $dir"
mkdir -p $dir

# 'file name' is the WAV_ID 
# e.g., A11_T648_HR1U_CH12_204-02-47-56_204-11-17-56-02 /scratch2/share/asj161130 \
# /Apollo_Corpora/Apollo11/A11_T648_HR1U_204-02-47-56_204-11-17-56 \
# /A11_T648_HR1U_CH12_204-02-47-56_204-11-17-56 \
# /A11_T648_HR1U_CH12_204-02-47-56_204-11-17-56-02.wav
find -L $audio_dir -name "*.wav" ! -name ".*" ! -name "*CH1_*" | \
  perl -ne '{ 
    chomp;
    $path=$_;
    next unless $path;
    @F = split"/", $path;
    ($f = $F[@F-1]) =~ s/.wav//;
    print "$f sox -R -t wav $path -t wav - rate 16000 dither | \n"
}' | sort > $dir/wav.scp

## [Option1] prepare segments file
#find -L $segment_dir -name "*.txt" | \
#  perl -ne '{
#    chomp;
#    $path = $_;
#    next unless $path;
#    @F = split"/", $path;
#    ($file_name = $F[-1]) =~ s/\.txt$//;
#    $count = 1;
#    open($fh, "<", $path) or die "Cannot open file $path: $!";
#    while (<$fh>) {
#        chomp;
#        my ($start, $end) = split /\s+/, $_;
#        next unless defined $start && defined $end;
#        print "$file_name\_$count $file_name $_\n";
#        $count++;
#    }
#    close($fh);
#}' | sort > $dir/segments

# [Option2] prepare segments file with 0.1 < segment < 60 seconds
folder_name=$(basename $audio_dir | cut -d'_' -f-2) # Get Axx_Txxx
find -L $segment_dir -iname "$folder_name*.txt" | \
  perl -ne '{
    chomp;
    $path = $_;
    next unless $path;
    @F = split"/", $path;
    ($file_name = $F[-1]) =~ s/\.txt$//;
    $count = 1;
    open($fh, "<", $path) or die "Cannot open file $path: $!";
    while (<$fh>) {
        chomp;
        my ($start, $end) = split /\s+/, $_;
        next unless defined $start && defined $end;
        next if $end - $start < 0.1;
        if ($end - $start > 60) {
            my $chunks = int(($end - $start) / 58) + 1;
            my $overlap = 2;
            my $chunk_start = $start;
            for (my $i = 0; $i < $chunks; $i++) {
                my $chunk_end = $chunk_start + 60;
                if ($chunk_end > $end || ($end - $chunk_end < 10)) {
                    $chunk_end = $end
                }
                if ($chunk_end - $chunk_start >= 0.1) {
                    my $c_id = $i + 1;
                    print "$file_name\_$count\_chunk$c_id $file_name $chunk_start $chunk_end\n";
                }
                # Stop if we have reached the end
                last if $chunk_end == $end;
                $chunk_start += 60 - $overlap;
            }
            $count++;
        } else {
            print "$file_name\_$count $file_name $_\n";
            $count++;
        }
    }
    close($fh);
}' | sort > $dir/segments

# prepare 'utt2spk' and 'spk2utt'
cut -d' ' -f1 $dir/segments > $dir/utt_id.tmp
paste -d ' ' $dir/utt_id.tmp $dir/utt_id.tmp | sort > $dir/utt2spk

utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

# generate empty text
sed -e 's/$/ a/' $dir/utt_id.tmp | sort > $dir/text

$cleanup && rm -f $dir/*.tmp

echo "$0 Finished preparing $dir"

