#!/bin/bash

# --- CONFIGURATION ---
BASE_DIR="/gscratch/tial/andysu/data"
# DATASETS=("Hindi_eng/beerbicep_Authors" "Hindi_eng/beerbicep_Mental_Physical" "Hindi_eng/beerbicep_Motivation" "Hindi_eng/beerbicep_science_technology" "Hindi_eng/beerbicep_Spirituality" "Hindi_eng/ranveer_creators")  # add manually Hindi 1
# DATASETS=("Hindi_eng/beerbicep_bollywood" "Hindi_eng/indian_silicon_valley" "Hindi_eng/ranveer_sports" "Hindi_eng/ranveer_veterans") # add manually Hindi 2
# DATASETS=("Hindi_eng/limitless_ishan" "Hindi_eng/singh_in_us" "Hindi_eng/singh_study_abroad" "Hindi_eng/TBWS_season3" "Hindi_eng/TBWS_season4") # add manually Hindi 3
# DATASETS=("UK_eng/book_author_story" "UK_eng/dr_sheen" "UK_eng/ceocast" "UK_eng/great_company" "UK_eng/FC_bullard" "UK_eng/we_talk") # add manually UK 1
DATASETS=("UK_eng/spurs_chat" "UK_eng/high_performance") # add manually UK 2
# DATASETS=("UK_eng/task_master" "UK_eng/frankie_lee") # add manually UK 3
SCRIPT_DIR="/gscratch/tial/andysu/TTS/speech_segmentation"
# --- END OF CONFIGURATION ---

# --- CONTROL SWITCHES ---
run_vad=1
run_sep_filter=1
run_diarization=1
run_rematch=1
run_transcribe=0
# --- END OF SWITCHES ---

# --- LOOP OVER DATASETS ---
for dataset in "${DATASETS[@]}"; do
    echo -e "\n===================="
    echo "Processing dataset: $dataset"
    echo "===================="

    DATASET_DIR="${BASE_DIR}/${dataset}"
    AUDIO_DIR="${DATASET_DIR}/audio"

    cd "$SCRIPT_DIR"
    source /gscratch/tial/junkaiwu/miniconda3/etc/profile.d/conda.sh

    # Step 1: VAD
    if [ "$run_vad" -eq 1 ]; then
        echo "--- Running VAD ---"
        conda activate /gscratch/tial/junkaiwu/miniconda3/envs/AudioPipeline
        python vad_multi.py --input_folder_path "$AUDIO_DIR" --num_threads 8 --sort
    fi

    # Step 2: Separation Filter
    if [ "$run_sep_filter" -eq 1 ]; then
        echo "--- Running Separation Filter ---"
        conda activate /gscratch/tial/junkaiwu/miniconda3/envs/AudioPipeline
        # python separation_filter.py --input_folder_path "$AUDIO_DIR" --skip_if_exist
        python separation_filter.py --input_folder_path "$AUDIO_DIR" 
    fi

    # Step 3: Diarization
    if [ "$run_diarization" -eq 1 ]; then
        echo "--- Running Diarization ---"
        conda deactivate
        conda activate /gscratch/tial/junkaiwu/miniconda3/envs/nemo
        python diarization.py --input_folder_path "$AUDIO_DIR" --v_r_threshold 0.995 --nv_r_threshold 0.0005 --silence_energy_threshold 0.0001 --silence_peak_threshold 0.001 
    fi

    # Step 4: Rematching
    if [ "$run_rematch" -eq 1 ]; then
        echo "--- Running Rematch ---"
        conda deactivate
        source /gscratch/tial/andysu/miniconda3/etc/profile.d/conda.sh
        conda activate /gscratch/tial/andysu/AAE_TTS
        python speaker_rematch.py --input_folder_path "$AUDIO_DIR"
    fi

    # Step 5: Transcription
    if [ "$run_transcribe" -eq 1 ]; then
        echo "--- Running Transcription ---"
        conda deactivate
        source /gscratch/tial/andysu/miniconda3/etc/profile.d/conda.sh
        conda activate /gscratch/tial/andysu/nemo_parakeet
        python transcribe_pipeline.py --base_search_dir "$DATASET_DIR" --output_base /gscratch/tial/andysu/TTS_samples --dnsmos_model_path /gscratch/tial/andysu/TTS/speech_segmentation/models/sig_bak_ovr.onnx --device cuda 
    fi

    echo "--- Finished processing: $dataset ---"
done

echo -e "\n[ALL finished] All datasets completed!"

