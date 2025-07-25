import os
import json
import numpy as np
import glob
import argparse
import time
import tqdm
import librosa
import soundfile as sf

import torch

from utils import *

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, labels_to_pyannote_object
from nemo.collections.asr.parts.utils.vad_utils import load_postprocessing_from_yaml
from huggingface_hub import get_token as get_hf_token

from omegaconf import OmegaConf

from collections import defaultdict

def get_single_overlapping_speaker_segments(diarization):

    # Parse segments
    segments = []
    for line in diarization[0]:
        start, end, speaker = line.split()
        segments.append((float(start), float(end), speaker))

    # Create timeline events
    events = []
    for start, end, speaker in segments:
        events.append((start, 'start', speaker))
        events.append((end, 'end', speaker))

    # Sort by time (with 'end' before 'start' if same time)
    events.sort(key=lambda x: (x[0], x[1] == 'start'))

    # Sweep line to track active speakers
    active_speakers = set()
    last_time = None
    results = []

    for time, typ, speaker in events:
        if last_time is not None and time != last_time:
            if active_speakers:
                results.append({
                    'start': last_time,
                    'end': time,
                    'speakers': sorted(active_speakers)
                })
        if typ == 'start':
            active_speakers.add(speaker)
        else:
            active_speakers.remove(speaker)
        last_time = time

    # Split into single and overlapping segments
    single_speaker_segments = [r for r in results if len(r['speakers']) == 1]
    overlapping_segments = [r for r in results if len(r['speakers']) > 1]

    return single_speaker_segments, overlapping_segments

def init_diar(model_path="models/diar_sortformer_4spk-v1.nemo"):
    diar_model = SortformerEncLabelModel.restore_from(restore_path=model_path, map_location=torch.device('cuda'), strict=False)
    return diar_model

def diarize(model, audio_path, parameters_path=None):
    pred_list, pred_tensor_list = model.diarize(audio=audio_path, postprocessing_yaml=parameters_path, include_tensor_outputs=True)

    single_speaker_segments, overlapping_segments = get_single_overlapping_speaker_segments(pred_list)
    return single_speaker_segments, overlapping_segments, pred_list, pred_tensor_list

def diarize_long_audio(
    model, 
    diarization_parameters_path,
    audio_path,
    v_r_threshold, 
    nv_r_threshold,
    silence_energy_threshold,
    silence_peak_threshold,
    interval=5.0, 
    max_duration=1000.0, 
    default_silence_duration=0.25,
    write_output=True,
    delete_temp=True,
    skip_if_exist=False
):
    if skip_if_exist:
        output_dir = get_output_dir(audio_path, "diarization")
        diarization_single_timestamps_path = f"{output_dir}/single_timestamps.json"
        diarization_overlapping_timestamps_path = f"{output_dir}/overlapping_timestamps.json"
        if os.path.isfile(diarization_single_timestamps_path) and os.path.isfile(diarization_overlapping_timestamps_path):
            print("Skipped exisiting diarization results...")
            diarization_single_timestamps = load_json(diarization_single_timestamps_path)
            diarization_overlapping_timestamps = load_json(diarization_overlapping_timestamps_path)
            return diarization_single_timestamps, diarization_overlapping_timestamps

    filter_output_dir = get_output_dir(audio_path, "separation_filter")

    vad_timestamps_path = f"{filter_output_dir}/vad_timestamps_window_3.0_hop_1.0.json"
    nonvad_timestamps_path = f"{filter_output_dir}/nonvad_timestamps.json"

    vad_timestamps = load_json(vad_timestamps_path)
    nonvad_timestamps = load_json(nonvad_timestamps_path)

    combined_timestamps, vocal_timestamps, silence_timestamps = combine_vocal_timestamps(
                                                                    vad_timestamps, 
                                                                    nonvad_timestamps, 
                                                                    v_r_threshold, 
                                                                    nv_r_threshold,
                                                                    silence_energy_threshold,
                                                                    silence_peak_threshold,  
                                                                    interval=interval, 
                                                                    max_duration=max_duration, 
                                                                    default_silence_duration=default_silence_duration
                                                                )
    # TODO: better handle no voice cases, currently this results in a failed diarization
    if len(vocal_timestamps) == 0:
        print("No clean voice in this file, skipping...")
        return [], []
    
    audio, sr = librosa.load(audio_path, sr=None)
    time2temppath = store_temp_audio(audio, sr, audio_path, combined_timestamps, "diarization")

    diarization_single_timestamps = []
    diarization_overlapping_timestamps = []

    print(f"Running Diarization on {len(combined_timestamps)} audio files...")

    for idx, ts in enumerate(combined_timestamps):
        s, e = get_se(ts)
        temppath = time2temppath[(s, e)]
        diarization_outputs = diarize(model, temppath, parameters_path=diarization_parameters_path)
        single_speaker_segments, overlapping_segments, pred_list, pred_tensor_list = diarization_outputs
        for _ in single_speaker_segments:
            s_, e_ = get_se(_)
            speaker = _["speakers"][0]
            assert len(_["speakers"]) == 1
            diarization_single_timestamps.append({"start": s_ + s, "end": e_ + s, "speaker": speaker, "idx": idx})
        for _ in overlapping_segments:
            s_, e_ = get_se(_)
            speakers = _["speakers"]
            diarization_overlapping_timestamps.append({"start": s_ + s, "end": e_ + s, "speakers": speakers, "idx": idx})
    
    if write_output:
        output_dir = get_output_dir(audio_path, "diarization")
        write_json(diarization_single_timestamps, f"{output_dir}/single_timestamps.json")
        write_json(diarization_overlapping_timestamps, f"{output_dir}/overlapping_timestamps.json")
    
    if delete_temp:
        delete_temp_audio(time2temppath)
    
    return diarization_single_timestamps, diarization_overlapping_timestamps

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--diarization_parameters_path",
        type=str,
        default="models/sortformer_diar_4spk-v2.yaml"
    )

    # for diarization
    parser.add_argument(
        "--v_r_threshold",
        type=float,
        default=0.98
    )
    parser.add_argument(
        "--nv_r_threshold",
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--silence_energy_threshold",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--silence_peak_threshold",
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=1000.0
    )
    parser.add_argument(
        "--default_silence_duration",
        type=float,
        default=0.25
    )
    parser.add_argument(
        "--skip_if_exist",
        action="store_true",
    )

    
    args = parser.parse_args()

    input_folder_path = args.input_folder_path
    diarization_parameters_path = args.diarization_parameters_path
    v_r_threshold = args.v_r_threshold
    nv_r_threshold = args.nv_r_threshold
    silence_energy_threshold = args.silence_energy_threshold
    silence_peak_threshold = args.silence_peak_threshold
    interval = args.interval 
    max_duration= args.max_duration
    default_silence_duration = args.default_silence_duration
    skip_if_exist = args.skip_if_exist

    diar_model = init_diar()

    audio_paths = get_audio_paths(input_folder_path)
    num_audio = len(audio_paths)

    print(f"Running Diarization on {num_audio} files...")
    failed = 0

    for audio_path in tqdm.tqdm(audio_paths):
        try:
            diarization_single_timestamps, diarization_overlapping_timestamps = diarize_long_audio(
                                                                                    diar_model, 
                                                                                    diarization_parameters_path,
                                                                                    audio_path,
                                                                                    v_r_threshold, 
                                                                                    nv_r_threshold,
                                                                                    silence_energy_threshold,
                                                                                    silence_peak_threshold,
                                                                                    interval=interval, 
                                                                                    max_duration=max_duration, 
                                                                                    default_silence_duration=default_silence_duration,
                                                                                    write_output=True,
                                                                                    delete_temp=True,
                                                                                    skip_if_exist=skip_if_exist
                                                                                )
        except:
            failed += 1
            continue
    completed = num_audio - failed
    print(f"Completed Diarization on {completed} files, failed on {failed} files.")
    

    

