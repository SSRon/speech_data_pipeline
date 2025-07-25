import os
import json
import tqdm
import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from utils import *
import time

def get_audio_duration(audio_path):
    try:
        return librosa.get_duration(filename=audio_path)
    except Exception:
        return 0  # or 0 if you want to treat broken ones as shortest

def init_silero():
    return load_silero_vad()

def silero_infer(model, audio_path, padding, write_output=False, skip_if_exist=False):
    if skip_if_exist:
        output_dir = get_output_dir(audio_path, "vad")
        vad_path = f"{output_dir}/timestamps.json"
        if os.path.isfile(vad_path):
            return None  # Already exists, skip
            
    wav = read_audio(audio_path)
    speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)

    audio_duration = librosa.get_duration(filename=audio_path)

    for ts in speech_timestamps:
        ts["start"] = max(ts["start"] - padding, 0.0)
        ts["end"] = min(ts["end"] + padding, audio_duration)

    speech_timestamps = combine_timestamps(speech_timestamps, interval=0.0, max_duration=99999999999.0)

    if write_output:
        output_dir = get_output_dir(audio_path, "vad")
        os.makedirs(output_dir, exist_ok=True)
        vad_path = f"{output_dir}/timestamps.json"
        with open(vad_path, 'w') as f:
            json.dump(speech_timestamps, f, indent=4)

    return speech_timestamps

def process_audio(audio_path, padding, skip_if_exist):
    try:
        model = load_silero_vad()
        silero_infer(model, audio_path, padding, write_output=True, skip_if_exist=skip_if_exist)
        return True
    except Exception:
        return False

if __name__ == "__main__":

    s = time.time()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder_path", type=str, required=True)
    parser.add_argument("--padding", type=float, default=0.1)
    parser.add_argument("--skip_if_exist", action="store_true")
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--sort", action="store_true")
    args = parser.parse_args()

    input_folder_path = args.input_folder_path
    padding = args.padding
    skip_if_exist = args.skip_if_exist
    num_threads = args.num_threads
    

    # model = init_silero()
    audio_paths = get_audio_paths(input_folder_path)

    if args.sort:
        audio_durations = []
        audio_paths_ = []
        for p in audio_paths:
            if skip_if_exist:
                output_dir = get_output_dir(p, "vad")
                vad_path = f"{output_dir}/timestamps.json"
                if os.path.isfile(vad_path):
                    continue
            
            duration = get_audio_duration(p)
            if duration == 0:
                continue
            
            audio_durations.append(duration)
            audio_paths_.append(p)

            order = np.argsort(audio_durations)[::-1]
            audio_paths = [audio_paths_[i] for i in order]
        
    print(f"Running VAD on {len(audio_paths)} files with {num_threads} threads...")

    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_audio, path, padding, skip_if_exist)
            for path in audio_paths
        ]

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                completed += 1
            else:
                failed += 1

    e = time.time()

    used_time = (e - s) / 60

    print(f"Completed VAD on {completed} files, failed on {failed} files, used {used_time} minutes")
