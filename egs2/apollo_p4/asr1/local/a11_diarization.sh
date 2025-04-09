#!/usr/bin/env bash

# Copyright 2023 University of Texas at Dallas (Szu-Jui Chen)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# This script is based on ESPnet. It will prepare the data, dumup audio, and inference.

set -e
set -u
set -o pipefail

min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# options
audio_dir=/scratch/sxc200004/pipeline_working/xxx #old location:/scratch2/share/asj161130/Apollo_Corpora/Apollo11/
segment_dir=/scratch/share/asj161130/sad_decisions/postproc_A11_thr0.123_nsp0.25/
dataset=
cleanup=true
gpu_inference=true
use_lm=true
dumpdir=dump/raw/
asr_exp="exp_ssl_mix/asr_train_asr_conformer_s3prlfrontend_wavlm_lr0.001_warm2w5_raw_en_bpe500_sp/"
lm_exp="exp_ssl_mix/lm_train_lm_transformer11_en_bpe500/"
inference_asr_model=valid.acc.ave.pth
inference_config="conf/decode.yaml"
inference_lm=valid.loss.ave.pth
inference_args=
nj=36 # for format_wav_scp.sh
inference_nj=8
assigned_gpu=

# if any job fail, we can specify the No. of scp for re-doing. For example, --redo "2 5 6 8"
# Or we can first run with --inference_nj X, where X > available GPUs, to get X splits of scps.
# Then using the redo to process by groups like --redo "1 2 3 4" --redo "5 6 7 8" --redo "9 10"
redo=

feats_type=raw
fs=16k
stage=1
stop_stage=100

. utils/parse_options.sh

log "$0 $*"
if [ $# -ne 0 ] ; then
  log "Error: No positional arguments are required"
  exit 1
fi
# Set the dataset to all folders as default
if [[ -z "$dataset" ]]; then
  dataset="$audio_dir/*"
fi

. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ -z "${redo}" ]; then
    log "Stage 1: Create data folders"
    # first get the list of empty segment files. The find with -empty does not work anymore due
    # to spaces/tab in the empty files. But since empty SAD files will not show in the segments.
    # So fix_data_dir.sh will remove the corresponding wav file from wav.scp.
    find -L $segment_dir -type f -name "*.txt" -empty > empty_sad.tmp
    while IFS= read -r line; do echo "$(basename "$line" .txt)"; done < empty_sad.tmp > empty_sad_filename.tmp

    # create data folder for each tape
    for folder in $dataset; do
        folder_name=$(basename "$folder")
        log "local/prepare_a11_data.sh --cleanup $cleanup , preparing folder $folder"
        local/prepare_a11_data.sh --cleanup $cleanup ${audio_dir}/${folder_name} ${segment_dir} data/${folder_name}
        
        # remove from wav.scp if the corresponding segment file is empty
        mv data/${folder_name}/wav.scp data/${folder_name}/wav.scp.tmp
        grep -v -f empty_sad_filename.tmp data/${folder_name}/wav.scp.tmp > data/${folder_name}/wav.scp
        rm -f data/${folder_name}/wav.scp.tmp
        utils/fix_data_dir.sh data/${folder_name}
        utils/validate_data_dir.sh --no-feats data/${folder_name} || exit 1
        utils/data/get_utt2dur.sh data/${folder_name}
    done

    # remove tmp files
    $cleanup && rm -f *.tmp
else
    log "Redo scps, skip stage 1 create data folder."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # The stage 3 of asr.sh in ESPnet
    log "Stage 2: Format wav.scp: data/ -> dump/raw/ and Decoding"
    for folder in $dataset; do
        dset=$(basename "$folder")
        expname=$(echo $dset | cut -d'_' -f1)
        if [ ! -f "${dumpdir}/${dset}/.done" ]; then
            ### dumping audio segments ###
            log "Dumping audio segments of $dset"
            utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${dumpdir}/${dset}"
            rm -f ${dumpdir}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}

            _opts=
            if [ -e data/"${dset}"/segments ]; then
                _opts+="--segments data/${dset}/segments "
            fi
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${cuda_cmd}" \
                --audio-format "wav" --fs "16000" ${_opts} \
                "data/${dset}/wav.scp" "${dumpdir}/${dset}"
            echo "${feats_type}" > "${dumpdir}/${dset}/feats_type"
            touch "${dumpdir}/${dset}/.done"
        else
            log "${dumpdir}/${dset}/.done exists. Dumping audio skipped"
        fi

        ### decoding ###
        log "Decoding $dset"
        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        fi
        inference_bin_tag=""
        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi
        if "${use_lm}"; then
            _opts+="--lm_train_config ${lm_exp}/config.yaml "
            _opts+="--lm_file ${lm_exp}/${inference_lm} "
        fi
        _data="${dumpdir}/${dset}"
        _dir="${asr_exp}/decode_${expname}_gpu_inference/${dset}"
        _logdir="${_dir}/logdir"
        _scp=wav.scp
        _type=sound
        mkdir -p "${_logdir}"

        if [ -z "${redo}" ]; then
            # 1. split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")

            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            utils/split_scp_spkdur.pl --utt2spk=data/${dset}/utt2spk --utt2dur=data/${dset}/utt2dur "${key_file}" ${split_scps}
        else
            log "Redo scps. Skip splitting scp..."
            _nj=$(echo "${redo}" | wc -w)
        fi

        # 2. submit decoding jobs
        log "Decoding started... log: '${_logdir}/asr_inference.*.log'"
        # Roughly estimate the working time with RTF 0.0715 when using 8 GPUs
        total_utt_dur=$(awk '{sum += $2} END {print sum}' ${dumpdir}/${dset}/utt2dur)
        time_estimate=$(awk -v total="$total_utt_dur" 'BEGIN {print total * 0.0715 / 3600}')
        log "Estimated to finish in $time_estimate hours."
        if [ -z "${redo}" ]; then
            rm -f "${_logdir}/*.log"
        fi

        if [ -z "${assigned_gpu}" ]; then
            # get available gpus
            # old method: available_gpus=($(nvidia-smi -L | cut -d':' -f1 | cut -d' ' -f2))
            total_gpus=$(nvidia-smi --list-gpus | wc -l)
            
            # Array to store the available GPUs
            available_gpus=()
            
            # Iterate over each GPU and check its status
            for ((gpu=0; gpu<total_gpus; gpu++)); do
                # Check if the GPU is in use, below is the old method:
                # if ! nvidia-smi -i "${gpu}" | grep "No running processes" > /dev/null; then
                #   countinue
                # fi
                # available_gpus+=("${gpu}")
                if ! nvidia-smi -i "${gpu}" | grep -i "python" > /dev/null; then
                    # Add the available GPU to the array
                    available_gpus+=("${gpu}")
                    if [ "${#available_gpus[@]}" -eq "${_nj}" ]; then
                        break # Exit the loop if the desired number of available GPUs is reached
                    fi
                fi
            done
        else
            available_gpus=($assigned_gpu)
        fi 
        if [[ ${#available_gpus[@]} -lt $_nj ]]; then
            echo "Error: Insufficient number of available GPUs ${#available_gpus[@]} for $_nj jobs."
            exit 1
        fi

        # start the Python jobs with CUDA_VISIBLE_DEVICES set for each GPU
        if [ -z "${redo}" ]; then
            for ((i=0; i<$_nj; i++)); do
                # shellcheck disable=SC2046,SC2086
                CUDA_VISIBLE_DEVICES=${available_gpus[i]} ${_cmd} --gpu "${_ngpu}" JOB=$(($i+1)) "${_logdir}"/asr_inference.JOB.log \
                    python3 -m espnet2.bin.asr_inference${inference_bin_tag} \
                        --batch_size 1 \
                        --ngpu "${_ngpu}" \
                        --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                        --key_file "${_logdir}"/keys.JOB.scp \
                        --asr_train_config "${asr_exp}"/config.yaml \
                        --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
                        --output_dir "${_logdir}"/output.JOB \
                        ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/asr_inference.*.log) ; exit 1; } &
            done
            wait
        else
            log "Re-do No. ${redo} scps..."
            ctemp=counter.tmp
            echo 0 > $ctemp
            for i in ${redo}; do
                count=$[$(cat $ctemp)]
                CUDA_VISIBLE_DEVICES=${available_gpus[count]} ${_cmd} --gpu "${_ngpu}" JOB=$i "${_logdir}"/asr_inference.JOB.log \
                    python3 -m espnet2.bin.asr_inference${inference_bin_tag} \
                        --batch_size 1 \
                        --ngpu "${_ngpu}" \
                        --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                        --key_file "${_logdir}"/keys.JOB.scp \
                        --asr_train_config "${asr_exp}"/config.yaml \
                        --asr_model_file "${asr_exp}"/"${inference_asr_model}" \
                        --output_dir "${_logdir}"/output.JOB \
                        ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/asr_inference.*.log) ; exit 1; } &
                count=$[$(cat $ctemp) + 1]
                echo $count > $ctemp
            done
            rm -f counter.tmp
            wait
        fi
        # check if all scp jobs are finished before next step
        finished_jobs=$(grep -l "(code 0)" ${_logdir}/asr_inference.*.log | wc -l || true)
        if [[ $finished_jobs -ne ${inference_nj} ]]; then
            log "Finished jobs: ${finished_jobs} not equal to total inference_nj ${inference_nj}. Please use redo option to finish all the jobs first before next step."
            exit 1
        fi

        # There will be an assertion error due to different number of start time marker
        # "INFO: speech length" and end time marker "INFO: best hypo". To use it, we will
        # need to modify the calculate_rtf.py
        # 3. calculate and report RTF based on decoding logs
        #log "Calculating RTF & latency... log: '${_logdir}/calculate_rtf.log'"
        #rm -f "${_logdir}"/calculate_rtf.log
        #_fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
        #_sample_shift=$(python3 -c "print(1 / ${_fs} * 1000)") # in ms
        #${_cmd} JOB=1 "${_logdir}"/calculate_rtf.log \
        #    calculate_rtf.py \
        #        --log-dir ${_logdir} \
        #        --log-name "asr_inference" \
        #        --input-shift ${_sample_shift} \
        #        --start-times-marker "speech length" \
        #        --end-times-marker "best hypo" \
        #        --inf-num 1
        
        # 4. concatenates the output files from each jobs
        # shellcheck disable=SC2068
        log "Gathering results from output folders..."
        for f in token token_int score text; do
            if [ -f "${_logdir}/output.1/1best_recog/${f}" ]; then
                for i in $(seq "${inference_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | sort -k1 >"${_dir}/${f}"
            fi
        done

        # 5. remove dumped audio to save space
        log "Removing dumped audio to save space..."
        rm -rf ${dumpdir}/${dset}/
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Post-processing ASR outputs."
    for folder in $dataset; do
        dset=$(basename "$folder")
        expname=$(echo $dset | cut -d'_' -f1)
        # remote <sos/eos> from transcriptions
        if [ ! -f "${asr_exp}/decode_${expname}_gpu_inference/${dset}/removal.done" ]; then
            sed -i 's/<sos\/eos>//g' ${asr_exp}/decode_${expname}_gpu_inference/${dset}/text
            touch ${asr_exp}/decode_${expname}_gpu_inference/${dset}/removal.done
            log "Done removing <sos/eos> from text for ${dset}." 
        fi
        # gathering text into trs file format
        mkdir -p ${audio_dir}/../asr_processed/${dset}_transcripts
        python3 local/convert_to_trs.py data/${dset}/segments ${asr_exp}/decode_${expname}_gpu_inference/${dset}/text ${audio_dir}/../asr_processed/${dset}_transcripts
        # Finished all ASR pipeline processing, mv the folder to asr_processed
        mv ${audio_dir}/${dset} ${audio_dir}/../asr_processed/
    done
fi

