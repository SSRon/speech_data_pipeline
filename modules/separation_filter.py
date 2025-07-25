# continue from vad
# for voiced segments, determine if it is vocal or noisy
# for other segments, determine if it is (relative) silence, or if it is too long and we don't care


import os
import json
import numpy as np
import glob
import argparse
import time
import tqdm
import librosa
import time

import torch

from models import separate_fast

from utils import *

# 
def trim_or_pad(arr, length):
    if len(arr) < length:
        arr = np.pad(arr, (0, length - len(arr)))
    else:
        arr = arr[:length]

    return arr

# 
def source_separation(predictor, audio):
    """
    Separate the audio into vocals and non-vocals using the given predictor.

    Args:
        predictor: The separation model predictor.
        audio (str or dict): The audio file path or a dictionary containing audio waveform and sample rate.

    Returns:
        dict: A dictionary containing the separated vocals and updated audio waveform.
    """

    mix, rate = None, None

    if isinstance(audio, str):
        mix, rate = librosa.load(audio, mono=False, sr=44100)
    else:
        # resample to 44100
        rate = audio["sample_rate"]
        if rate != 44100:
            print("resample")
            mix = librosa.resample(audio["waveform"], orig_sr=rate, target_sr=44100)
        else:
            mix = audio["waveform"]

    vocals, no_vocals = predictor.predict(mix)

    # convert vocals back to previous sample rate
    # logger.debug(f"vocals shape before resample: {vocals.shape}")
    if rate != 44100:
        print("resample")
        vocals = librosa.resample(vocals.T, orig_sr=44100, target_sr=rate).T
    # logger.debug(f"vocals shape after resample: {vocals.shape}")

    audio_name = audio["name"]

    vocal_waveform = trim_or_pad(vocals[:, 0], len(audio["waveform"]))

    vocal = {"waveform": vocal_waveform, "name": f"{audio_name}_vocal", "sample_rate": rate}
    nonvocal = {"waveform": audio["waveform"] - vocal_waveform, "name": f"{audio_name}_nonvocal", "sample_rate": rate}

    return vocal, nonvocal

def init_separator(config_path="models/separation_config.json", device_name="cuda"):
    cfg = load_cfg(config_path)
    model = separate_fast.Predictor(args=cfg["separate"]["step1"], device=device_name)
    return model

def separator_infer(model, audio, sr=44100, return_waveform=True):
    mixture = {"waveform": audio, "name": None, "sample_rate": sr}
    vocal, nonvocal = source_separation(model, mixture)
    
    if return_waveform:
        return vocal["waveform"], nonvocal["waveform"]
    else:
        return vocal, nonvocal

def get_nonvad_timestamps(audio, sr, audio_path, vad_timestamps, write_output=True, skip_if_exist=False):

    if skip_if_exist:
        separation_filer_output_dir = get_output_dir(audio_path, "separation_filter")
        nonvad_timestamps_path = f"{separation_filer_output_dir}/nonvad_timestamps.json"
        if os.path.isfile(nonvad_timestamps_path):
            print("Skipped exisiting nonvad results...")
            nonvad_timestamps = load_json(f"{separation_filer_output_dir}/nonvad_timestamps.json")
            # TODO: make use of energy dict for faster running
            return nonvad_timestamps, None, audio, sr
    
    if audio is None:
        audio, sr = librosa.load(audio_path, sr=None)

    # avoid duplicate calculation
    energy_dict = {}

    # dealing with silence
    nonvad_timestamps = []
    for i in tqdm.tqdm(range(len(vad_timestamps)-1)):
        start_prev, end_prev = vad_timestamps[i]["start"], vad_timestamps[i]["end"]
        start_next, end_next = vad_timestamps[i+1]["start"], vad_timestamps[i+1]["end"]


        if start_next < end_prev:
            continue
        
        audio_prev = cut_audio(audio, sr, start_prev, end_prev)
        audio_next = cut_audio(audio, sr, start_next, end_next)
        # the audio we want to know if it's silence
        audio_ = cut_audio(audio, sr, end_prev, start_next)

        if len(audio_) == 0:
            continue

        # calculates ratios
        if (start_prev, end_prev) not in energy_dict:
            energy_prev = energy(audio_prev, take_mean=True)
            energy_dict[(start_prev, end_prev)] = energy_prev
        else:
            energy_prev = energy_dict[(start_prev, end_prev)]

        if (start_next, end_next) not in energy_dict:
            energy_next = energy(audio_next, take_mean=True)
            energy_dict[(start_next, end_next)] = energy_next
        else:
            energy_next = energy_dict[(start_next, end_next)]
        
        energy_ = energy(audio_, take_mean=True)

        peak = float(max(abs(audio_)))
        energy_ratio_prev = float(energy_ / energy_prev)
        energy_ratio_next = float(energy_ / energy_next)
        peak_ratio_prev = float(peak / max(abs(audio_prev)))
        peak_ratio_next = float(peak / max(abs(audio_next)))
        

        nonvad_timestamps.append({"start": end_prev, 
                                  "end": start_next,
                                  "energy": float(energy_),
                                  "peak": peak,
                                  "energy_ratio_prev": energy_ratio_prev,
                                  "energy_ratio_next": energy_ratio_next,
                                  "peak_ratio_prev": peak_ratio_prev,
                                  "peak_ratio_next": peak_ratio_next,
                                  "prev": {"start": start_prev, "end": end_prev},
                                  "next": {"start": start_next, "end": end_next}
                                 }
                                )
        
    if write_output:
        separation_filer_output_dir = get_output_dir(audio_path, "separation_filter")
        os.makedirs(separation_filer_output_dir, exist_ok=True)
        with open(f"{separation_filer_output_dir}/nonvad_timestamps.json", "w") as f:
            json.dump(nonvad_timestamps, f, indent=4)
    
    return nonvad_timestamps, energy_dict, audio, sr

def separate_long_audio(model, audio, sr, audio_path, timestamps, skip_separation_duration, max_separation_duration, print_info=False):
    timestamps_separation = combine_timestamps(timestamps, interval=skip_separation_duration, max_duration=max_separation_duration)

    original_num = len(timestamps)
    original_duration = np.sum([_["end"] - _["start"] for _ in timestamps])
    combined_num = len(timestamps_separation)
    combined_duration = np.sum([_["end"] - _["start"] for _ in timestamps_separation])

    if print_info:
        print(f"Original Number of Segments: {original_num}")
        print(f"Original Duration: {original_duration}")
        print(f"Combined Number of Segments: {combined_num}")
        print(f"Combined Duration: {combined_duration}")

    separation_results = {}
    for i, _ in enumerate(timestamps_separation):
        s, e = get_se(_)
        mixture = cut_audio(audio, sr, s, e)
        vocal, nonvocal = separator_infer(model, mixture, sr=sr)
        separation_results[i] = {"start": s, "end": e, "mixture": mixture, "vocal": vocal, "nonvocal": nonvocal}

    return separation_results

def find_segment(start, end, separation_results, idx):
    for i in range(idx, len(separation_results)):
        if start >= separation_results[i]["start"] and end <= separation_results[i]["end"]:
            return i


def separate_filter(model, audio, sr, audio_path, timestamps, skip_separation_duration, max_separation_duration, window_size=8.0, hop_size=3.0, debug_n=None, write_output=True, skip_if_exist=False):

    if skip_if_exist:
        separation_filer_output_dir = get_output_dir(audio_path, "separation_filter")
        vad_timestamps_path = f"{separation_filer_output_dir}/vad_timestamps_window_{window_size}_hop_{hop_size}.json"
        if os.path.isfile(vad_timestamps_path):
            print("Skipped exisiting vad results...")
            vad_timestamps = load_json(vad_timestamps_path)
            return vad_timestamps
    
    if audio is None:
        audio, sr = librosa.load(audio_path, sr=None)
    
    vad_timestamps = []

    # run a few for debugging
    if debug_n is None:
        n_ = len(timestamps)
    else:
        n_ = debug_n
    
    audio_end = timestamps[n_ - 1]["end"]
    
    separation_results = separate_long_audio(model, cut_audio(audio, sr, 0.0, audio_end), sr, audio_path, timestamps[:n_], skip_separation_duration, max_separation_duration, print_info=True)

    segment_idx = 0
    
    for _ in timestamps[:n_]:

        start = _["start"]
        end = _["end"]

        dur = end - start

        # find the separation result
        segment_idx = find_segment(start, end, separation_results, segment_idx)
        segment_mixture = separation_results[segment_idx]["mixture"]
        segment_vocal = separation_results[segment_idx]["vocal"]
        segment_nonvocal = separation_results[segment_idx]["nonvocal"]
        segment_start = separation_results[segment_idx]["start"]
        segment_end = separation_results[segment_idx]["end"]
        
        mixture = cut_audio(segment_mixture, sr, start - segment_start, end - segment_start)
        vocal = cut_audio(segment_vocal, sr, start - segment_start, end - segment_start)
        nonvocal = cut_audio(segment_nonvocal, sr, start - segment_start, end - segment_start)
        
        if dur <= window_size:
            # no sliding window
            v_r, nv_r = compute_ratio(vocal, nonvocal)
            vad_timestamps.append({"start": start, "end": end, "v_r": float(v_r), "nv_r": float(nv_r)})
        else:
            # sliding window
            n_segments = int(np.ceil((dur - window_size + hop_size) / hop_size))
            assert ((n_segments - 1) * hop_size + window_size) >= dur
            
            for n in range(n_segments):
                s_ = int(n * hop_size * sr)
                e_ = s_ + int(window_size * sr)

                v_r, nv_r = compute_ratio(vocal[s_:e_], nonvocal[s_:e_])

                vad_timestamps.append({"start": start + n * hop_size, "end": min(start + n * hop_size + window_size, end), "v_r": float(v_r), "nv_r": float(nv_r)})

                if (n * hop_size + window_size) >= dur:
                    break

    if write_output:
        separation_filer_output_dir = get_output_dir(audio_path, "separation_filter")
        os.makedirs(separation_filer_output_dir, exist_ok=True)
        write_json(vad_timestamps, f"{separation_filer_output_dir}/vad_timestamps_window_{window_size}_hop_{hop_size}.json")

    return vad_timestamps

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path",
        type=str,
        required=True,
    )

    
    # separation
    parser.add_argument(
        "--skip_separation_duration",
        type=float,
        default=5.0,
        help="a nonvad segments longer than this will be skipped"
    )
    parser.add_argument(
        "--max_separation_duration",
        type=float,
        default=60.0,
        help="pick a good duration"
    )
    parser.add_argument(
        "--window_size",
        type=float,
        default=3.0
    )
    parser.add_argument(
        "--hop_size",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--skip_if_exist",
        action="store_true",
    )

    args = parser.parse_args()

    input_folder_path = args.input_folder_path
    skip_separation_duration = args.skip_separation_duration
    max_separation_duration = args.max_separation_duration
    window_size = args.window_size
    hop_size = args.hop_size
    skip_if_exist = args.skip_if_exist

    model = init_separator(config_path="models/source_separation_config.json")

    audio_paths = get_audio_paths(input_folder_path)
    num_audio = len(audio_paths)

    print(f"Running Separation and Filtering on {num_audio} files...")
    failed = 0

    for audio_path in tqdm.tqdm(audio_paths):
        try:
            vad_output_dir = get_output_dir(audio_path, "vad")
            with open(f"{vad_output_dir}/timestamps.json", "r") as f:
                vad_timestamps = json.load(f)

            if len(vad_timestamps) == 0:
                print("No voice in this file, skip...")
                continue
            
            # audio, sr = librosa.load(audio_path, sr=None)

            # deal with nonvad segments     
            nonvad_timestamps, energy_dict, audio, sr = get_nonvad_timestamps(None, None, audio_path, vad_timestamps, write_output=True, skip_if_exist=skip_if_exist)

            # deal with vad segments
            vad_timestamps_ = separate_filter(
                model, 
                audio, 
                sr, 
                audio_path,
                vad_timestamps,
                skip_separation_duration, 
                max_separation_duration, 
                window_size=window_size, 
                hop_size=hop_size, 
                debug_n=None, 
                write_output=True,
                skip_if_exist=skip_if_exist
            )
        except:
            failed += 1
            continue
    completed = num_audio - failed
    print(f"Completed Separation on {completed} files, failed on {failed} files.")




        
        
        














