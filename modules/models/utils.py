import os
import json
import numpy as np
import glob
import argparse
import time
import tqdm
import soundfile as sf
import matplotlib.pyplot as plt

def get_audio_paths(path):
    files = []
    for ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']:
        files += glob.glob(f"{path}/*{ext}")

    return files

def remove_audio_extension(filename):
    for ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']:
        if filename.lower().endswith(ext):
            return filename[:-len(ext)]
    return filename

# from emilia
def load_cfg(cfg_path):
    """
    Load configuration from a JSON file.

    Args:
        cfg_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"{cfg_path} not found. Please copy, configure, and rename `config.json.example` to `{cfg_path}`."
        )
    with open(cfg_path, "r") as f:
        try:
            cfg = json.load(f)
        except json.decoder.JSONDecodeError as e:
            raise TypeError(
                "Please finish the `// TODO:` in the `config.json` file before running the script. Check README.md for details."
            )
    return cfg

def energy(x, take_mean=False):
    if take_mean:
        return np.mean(x**2)
    else:
        return np.sum(x ** 2)

def compute_ratio(v, nv):
    m = v + nv
    return energy(v) / energy(m), energy(nv) / energy(m)

def normalize(audio):
    audio -= np.mean(audio)
    max = np.max(abs(audio))
    audio /= max
    return audio * 0.95

def combine_timestamps(timestamps, interval=0.25, max_duration=10.0, must_include_voice=False):
    # for now type is one of voice and silence
    new_timestamps = []
    if len(timestamps) == 0:
        return []
    current_segment = None
    for segment in timestamps:
        if current_segment is None:
            current_segment = segment
            if must_include_voice:
                # add a types key
                assert "type" in current_segment
                current_segment["types"] = [current_segment["type"]]
        else:
            if (segment["start"] - current_segment["end"]) > interval or (segment["end"] - current_segment["start"]) > max_duration:
                # finished one combination
                if must_include_voice:
                    assert "types" in current_segment
                    if "voice" in current_segment["types"]:
                        new_timestamps.append(current_segment)
                    current_segment = segment
                    assert "type" in current_segment
                    current_segment["types"] = [current_segment["type"]]
                else:
                    new_timestamps.append(current_segment)
                    current_segment = segment
            else:
                if must_include_voice:
                    assert "types" in current_segment and "type" in segment
                    new_types = current_segment["types"] + [segment["type"]]
                    current_segment = {"start": current_segment["start"], "end": segment["end"], "types": new_types}
                else:
                    current_segment = {"start": current_segment["start"], "end": segment["end"]}
    if must_include_voice:
        if "voice" in current_segment["types"]:
            new_timestamps.append(current_segment)
    else:
        new_timestamps.append(current_segment)
    return new_timestamps

def combine_vocal_timestamps(
    vad_timestamps, 
    nonvad_timestamps, 
    v_r_threshold, 
    nv_r_threshold,
    silence_energy_threshold,
    silence_peak_threshold,  
    interval=1.0, 
    max_duration=10.0, 
    default_silence_duration=0.25
):
    # filter vad timestamps
    vocal_timestamps = []
    noisy_timestamps = []
    for _ in vad_timestamps:
        v_r = _["v_r"]
        nv_r = _["nv_r"]
        if v_r >= v_r_threshold and nv_r <= nv_r_threshold:
            vocal_timestamps.append(_)
        else:
            noisy_timestamps.append(_)
    
    # filter nonvad timestamps
    silence_timestamps = []
    for _ in nonvad_timestamps:
        s, e = get_se(_)
        if (e - s) < default_silence_duration:
            silence_timestamps.append(_)
        energy_ratio_prev = _["energy_ratio_prev"]
        energy_ratio_next = _["energy_ratio_next"]
        peak_ratio_prev = _["peak_ratio_prev"]
        peak_ratio_next = _["peak_ratio_next"]

        if (
            energy_ratio_prev < silence_energy_threshold
            and energy_ratio_next < silence_energy_threshold
            and peak_ratio_prev < silence_peak_threshold
            and peak_ratio_next < silence_peak_threshold   
        ):
            silence_timestamps.append(_)

    combined_timestamps = []
   
    timestamps = []
    for i, _ in enumerate(vocal_timestamps):
        timestamps.append({"type": "voice", "idx": i, "start": _["start"], "end": _["end"]})
    for i, _ in enumerate(silence_timestamps):
        timestamps.append({"type": "silence", "idx": i, "start": _["start"], "end": _["end"]})
    sorted_timestamps = sorted(timestamps, key=lambda _: _["start"])
    is_sorted = all(sorted_timestamps[i]["end"] <= sorted_timestamps[i + 1]["end"] for i in range(len(sorted_timestamps) - 1))

    if len(noisy_timestamps) == 0:
        # no need to worry putting noisy segments in
        combined_timestamps = combine_timestamps(sorted_timestamps, interval=interval, max_duration=max_duration, must_include_voice=True)
    else:
        noisy_idx = 0
        noisy_start, noisy_end = get_se(noisy_timestamps[0])
        sorted_timestamps_ = []
        # print(noisy_start)
        # print(noisy_end)
        # print(sorted_timestamps)
        for _ in sorted_timestamps:
            s, e = get_se(_)
            if e <= noisy_start:
                sorted_timestamps_.append(_)
            else:
                # combine!
                # print(sorted_timestamps_, noisy_start, noisy_end)
                combined_timestamps += combine_timestamps(sorted_timestamps_, interval=interval, max_duration=max_duration, must_include_voice=True)
                # update noisy idx
                sorted_timestamps_ = []
                while e > noisy_end:
                    noisy_idx += 1
                    assert noisy_idx <= len(noisy_timestamps)
                    if noisy_idx == len(noisy_timestamps):
                        noisy_start = 99999999999999999 # a non existing noisy segment
                        noisy_end = 99999999999999999 + 1 # a non existing noisy segment
                    else:
                        noisy_start, noisy_end = get_se(noisy_timestamps[noisy_idx])
                sorted_timestamps_.append(_)
        
    return combined_timestamps, vocal_timestamps, silence_timestamps

def cut_audio(audio, sr, start, end):
    return audio[int(start*sr):int(end*sr)]

def get_output_dir(audio_path, task):
    assert task in ["vad", "separation_filter", "diarization", "rematch"]
    dirname = os.path.dirname(audio_path)
    audioname = remove_audio_extension(os.path.basename(audio_path))
    output_dir = f"{dirname}/{audioname}_outputs/{task}"
    return output_dir

def load_json(json_path):
    with open(json_path, "r") as f:
        _ = json.load(f)
    return _

def write_json(data, json_path):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

def get_se(timestamp):
    return timestamp["start"], timestamp["end"]

def store_temp_audio(audio, sr, audio_path, timestamps, task):
    time2temp = {}
    time2temppath = {}
    
    for _ in timestamps:
        s, e = get_se(_)
        time2temp[(s, e)]= cut_audio(audio, sr, s, e)
    
    output_dir = get_output_dir(audio_path, task)
    os.makedirs(output_dir, exist_ok=True)

    for s, e in time2temp:
        sf.write(f"{output_dir}/temp_{s}_{e}.mp3", time2temp[(s,e)], sr)
        time2temppath[(s, e)] = f"{output_dir}/temp_{s}_{e}.mp3"

    return time2temppath

def delete_temp_audio(time2temppath):
    for s, e in time2temppath:
        temppath = time2temppath[(s, e)]
        os.system(f"rm {temppath}")

def plot_diarization(pred_tensor_list, speaker_indices, s, e, duration):
    pred_array_list = pred_tensor_list[0].numpy()[0]
    length = pred_array_list.shape[0]
    sframe = int(np.floor(s * length / duration))
    eframe = int(np.ceil(e * length / duration))
    for i in speaker_indices:
        plt.plot(pred_array_list[sframe:eframe, i], label=f"speaker_{i}")
    plt.legend()
    plt.show()



    


