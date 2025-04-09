#!/usr/bin/env bash

base=
proposed=
test_sets=

. utils/parse_options.sh

. ./path.sh

mkdir -p mapsswe 

for dset in ${test_sets}; do
    cp ${base}/${dset}/score_wer/ref.trn mapsswe/ref.trn
    cp ${base}/${dset}/score_wer/hyp.trn mapsswe/base.trn
    cp ${proposed}/${dset}/score_wer/ref.trn mapsswe/pref.trn
    cp ${proposed}/${dset}/score_wer/hyp.trn mapsswe/proposed.trn
    sclite -F -i rm -r mapsswe/ref.trn -h mapsswe/base.trn -o sgml
    sclite -F -i rm -r mapsswe/pref.trn -h mapsswe/proposed.trn -o sgml
    cat mapsswe/base.trn.sgml mapsswe/proposed.trn.sgml | sc_stats -p -t mapsswe -v -u -n mapsswe/result.mapsswe.${dset}
done


