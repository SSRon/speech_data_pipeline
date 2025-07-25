"""
python host_rematch.py --data_root="/gscratch/tial/andysu/data/Hindi_eng/beerbicep_Authors/"
python host_rematch.py --data_root="/gscratch/tial/andysu/data/Hindi_eng/beerbicep_Mental_Physical/"
python host_rematch.py --data_root="/gscratch/tial/andysu/data/Hindi_eng/beerbicep_Motivation/"
python host_rematch.py --data_root="/gscratch/tial/andysu/data/Hindi_eng/beerbicep_science_technology/"
python host_rematch.py --data_root="/gscratch/tial/andysu/data/Hindi_eng/beerbicep_Spirituality/"
python host_rematch.py --data_root="/gscratch/tial/andysu/data/Hindi_eng/indian_silicon_valley/"
python host_rematch.py --data_root="/gscratch/tial/andysu/data/Hindi_eng/ranveer_creators/"
python host_rematch.py --data_root="/gscratch/tial/andysu/data/Hindi_eng/ranveer_sports/"
python host_rematch.py --data_root="/gscratch/tial/andysu/data/Hindi_eng/ranveer_veterans/"
"""

import os
import json
from collections import defaultdict
from pydub import AudioSegment
from speechbrain.inference.speaker import SpeakerRecognition

def match_hosts(reference_dir, json_path, full_audio_path, max_duration_sec=180.0, device="cuda"):
    reference_signals = {}
    if not os.path.exists(reference_dir) or not os.listdir(reference_dir):
        return None, "No reference signals! Leave host matching."

    for file in os.listdir(reference_dir):
        if file.endswith(".wav"):
            host_id = os.path.splitext(file)[0]
            reference_signals[host_id] = os.path.join(reference_dir, file)

    if not reference_signals:
        return None, "No reference signals! Leave host matching."

    with open(json_path, "r") as f:
        timestamps = json.load(f)

    speaker_segments = defaultdict(list)
    speaker_durations = defaultdict(float)
    for item in timestamps:
        speaker = item["speaker"]
        start = item["start"]
        end = item["end"]
        duration = end - start
        speaker_segments[speaker].append((start, end))
        speaker_durations[speaker] += duration

    total_all_duration = sum(speaker_durations.values())
    sorted_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)

    # Determine which speakers to verify based on custom rules
    speakers_to_verify = []
    if len(speaker_durations) == 2:
        # Case 1: Exactly 2 speakers
        spk1, dur1 = sorted_speakers[0]
        spk2, dur2 = sorted_speakers[1]
        ratio = dur1 / dur2
        if ratio >= 3.0:
            # If one speaks 3x longer, match only the shorter one
            shorter = spk2
            proportion = dur2 / total_all_duration
            results = {}
            host_ids = list(reference_signals.keys())

            if len(host_ids) >= 1:
                results[host_ids[0]] = {
                    "host_total_duration": round(dur2, 2),
                    "host_proportion": round(proportion, 4),
                    "matched_speakers": [
                        {
                            "speaker": shorter,
                            "score": 1.0,
                            "prediction": True,
                            "audio_duration": round(dur2, 2)
                        }
                    ]
                }

            for host_id in host_ids[1:]:
                results[host_id] = {
                    "host_total_duration": 0.0,
                    "host_proportion": 0.0,
                    "matched_speakers": []
                }

            return results, None
        else:
            # If ratio < 3.0, verify both
            speakers_to_verify = [spk1, spk2]
    else:
        # Case 2: More than 2 speakers
        spk1, dur1 = sorted_speakers[0]
        rest_total = sum(dur for _, dur in sorted_speakers[1:])
        if dur1 >= 2 * rest_total:
            # If one speaker dominates, skip them
            speakers_to_verify = [s for s, _ in sorted_speakers[1:]]
        else:
            # Otherwise, verify all
            speakers_to_verify = [s for s, _ in sorted_speakers]

    audio = AudioSegment.from_file(full_audio_path).set_channels(1)
    speaker_audios = {}
    for speaker in speakers_to_verify:
        total = 0.0
        combined = AudioSegment.silent(duration=0)
        for start, end in sorted(speaker_segments[speaker], key=lambda x: x[1] - x[0], reverse=True):
            dur = end - start
            if total + dur > max_duration_sec:
                break
            combined += audio[int(start * 1000):int(end * 1000)]
            total += dur
        speaker_audios[speaker] = (combined, total)

    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )

    results = {}
    for host_id, host_path in reference_signals.items():
        results[host_id] = {
            "host_total_duration": 0.0,
            "matched_speakers": []
        }
        for speaker, (audio_clip, dur) in speaker_audios.items():
            temp_path = f"temp_{speaker}.wav"
            if len(audio_clip) == 0:
                print(f"[WARNING] Skipping {speaker} due to empty audio.")
                continue
            
            audio_clip.export(temp_path, format="wav")
            score, prediction = verification.verify_files(host_path, temp_path)
            os.remove(temp_path)

            score_val = score.item()
            prediction_val = prediction.item()

            if prediction_val and score_val > 0.5:
                results[host_id]["matched_speakers"].append({
                    "speaker": speaker,
                    "score": round(score_val, 4),
                    "prediction": True,
                    "audio_duration": round(dur, 2)
                })
                results[host_id]["host_total_duration"] += dur

        results[host_id]["host_total_duration"] = round(results[host_id]["host_total_duration"], 2)
        results[host_id]["host_proportion"] = round(
            results[host_id]["host_total_duration"] / total_all_duration, 4
        )

    return results, None


def batch_host_matching(data_root):
    reference_dir = os.path.join(data_root, "reference_signal")
    if not os.path.exists(reference_dir):
        print("No reference signals! Leave host matching.")
        return

    match_dir = os.path.join(data_root, "match_host")
    os.makedirs(match_dir, exist_ok=True)
    output_path = os.path.join(match_dir, "match_host.json")

    if os.path.exists(output_path):
        print("match_host.json already exists. Skipping host matching.")
        return

    results_dict = {}
    audio_dir = os.path.join(data_root, "audio")
    for file in os.listdir(audio_dir):
        if file.endswith(".mp3"):
            yt_id = os.path.splitext(file)[0]
            audio_path = os.path.join(audio_dir, file)
            json_path = os.path.join(audio_dir, f"{yt_id}_outputs", "rematch", "rematched_timestamps.json")

            if not os.path.exists(json_path):
                continue

            print(f"Processing {yt_id}...")
            results, msg = match_hosts(reference_dir, json_path, audio_path)

            if msg:
                print(f"  ↪ Skipped: {msg}")
                continue

            results_dict[yt_id] = results

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print("✔️ Host matching completed. Output saved to match_host/match_host.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch host speaker matching based on rematched diarization.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing audio/, reference_signal/, and *_outputs folders inside audio/")
    args = parser.parse_args()
    batch_host_matching(args.data_root)
