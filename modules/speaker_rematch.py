#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, argparse
import numpy as np
import torch, torchaudio
from tqdm import tqdm
import sys
from speechbrain.inference.classifiers import EncoderClassifier
from cuml import UMAP
from cuml.cluster import hdbscan as cuml_hdbscan
from cuml.manifold import TSNE
from utils import get_audio_paths, get_output_dir, load_json, write_json

DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
TARGET_SR = 16000          # resample target

def standardize_audio_format(signal, sr, target_sr=16000):
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        signal = resampler(signal)
    return signal.squeeze(0)

def combine_same_speaker(timestamps, interval=0.25, max_duration=20.0, min_duration=3.0):
    # for now type is one of voice and silence
    new_timestamps = []
    if len(timestamps) == 0:
        return []
    current_segment = None
    for segment in timestamps:
        if current_segment is None:
            current_segment = segment
        else:
            if (segment["start"] - current_segment["end"]) > interval or (segment["end"] - current_segment["start"]) > max_duration or segment["speaker"] != current_segment["speaker"]:
                # finished one combination
                new_timestamps.append(current_segment)
                current_segment = segment
            else:
                current_segment = {"start": current_segment["start"], "end": segment["end"], "speaker": current_segment["speaker"]}
    new_timestamps.append(current_segment)
    new_timestamps_ = [_ for _ in new_timestamps if _["end"] - _["start"] > min_duration]
    return new_timestamps_

def separate_audio(audio_path: str, timestamps: list,
                   target_sr: int = 16000, min_duration: float = None):
    signal, sr = torchaudio.load(audio_path)

    audio_chunks = []
    filtered_ts = []

    for ts in timestamps:
        duration = ts["end"] - ts["start"]
        if min_duration is not None and duration < min_duration:
            continue

        start_sample = int(ts["start"] * sr)
        end_sample = int(ts["end"] * sr)
        if end_sample > signal.shape[1]:
            continue

        chunk = signal[:, start_sample:end_sample]
        if chunk.shape[0] > 1 or sr != target_sr:
            chunk = standardize_audio_format(chunk, sr, target_sr)

        audio_chunks.append(chunk)
        filtered_ts.append(ts)

    return audio_chunks, filtered_ts

def encode_audio_chunks(audio_chunks, classifier):
    device = classifier.device
    total  = len(audio_chunks)
    print(f"Encoding {total} chunks on {device}")
    embeddings = []

    for idx, chunk in tqdm(
        enumerate(audio_chunks, 1),
        total=total,
        desc="  • encoding",
        leave=False,
    ):
        chunk = chunk.to(device)
        with torch.no_grad():
            emb = classifier.encode_batch(chunk.unsqueeze(0))
        embeddings.append((idx, emb.squeeze(0).cpu()))

    return embeddings

def embeddings_clustering(embeddings, n_components=5, min_cluster_ratio=0.04, soft_threshold=0.2):
    print(f"Converting {len(embeddings)} embeddings to numpy...")
    X = np.stack([emb.squeeze(0).cpu().numpy().astype(np.float32) for _, emb in embeddings])

    reduced = None
    try:
        print(f"Running UMAP on GPU with n_components={n_components}...")
        umap = UMAP(n_components=n_components, random_state=42)
        reduced = umap.fit_transform(X)
        print(f"[DEBUG] UMAP reduced shape: {reduced.shape}")
        if np.isnan(reduced).any() or np.isinf(reduced).any():
            raise ValueError("UMAP output contains NaN or Inf.")

    except Exception as e:
        print(f"[WARNING] UMAP failed: {e}")
        try:
            print(f"Falling back to TSNE with n_components=2...")
            tsne = TSNE(n_components=2, random_state=42)
            reduced = tsne.fit_transform(X)
            print(f"[DEBUG] TSNE reduced shape: {reduced.shape}")
            if np.isnan(reduced).any() or np.isinf(reduced).any():
                raise ValueError("TSNE output contains NaN or Inf.")
        except Exception as tsne_e:
            print(f"[FAIL] TSNE also failed: {tsne_e}")
            print(f"[INFO] Skipping this sample due to UMAP and TSNE failure.")
            return None, None, None  # give up current sample

    min_cluster_size = max(2, int(len(embeddings) * min_cluster_ratio))
    print(f"Running cuML HDBSCAN clustering (min_cluster_size={min_cluster_size})...")
    clusterer = cuml_hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
    clusterer.fit(reduced)

    print("Getting soft cluster membership probabilities...")
    soft_clusters = cuml_hdbscan.all_points_membership_vectors(clusterer)
    labels = clusterer.labels_

    print(f"Initial clustering complete. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")

    labels_fixed = labels.copy()
    for i in range(len(labels)):
        if labels[i] == -1:
            max_prob = soft_clusters[i].max()
            best_cluster = soft_clusters[i].argmax()
            if max_prob >= soft_threshold:
                labels_fixed[i] = best_cluster

    print(f"Final cluster assignment complete. Remaining -1 labels: {(labels_fixed == -1).sum()}")
    return labels_fixed, reduced, soft_clusters



def assign_labels(ts, labels):
    assert len(ts) == len(labels)
    for i, lab in enumerate(labels):
        ts[i]["speaker"] = f"speaker_{lab}" if lab != -1 else "unknown"
    return ts

def is_within_combined_and_rematch(segment, combined_segments):
    """Check if a segment is fully within any combined segment, and return rematched speaker if found"""
    for comb in combined_segments:
        if (
            segment["start"] >= comb["start"]
            and segment["end"] <= comb["end"]
        ):
            return comb["speaker"]  # Use rematched speaker ID
    return None

def process_one_audio(audio_path, skip_if_exist, n_components, keep_diar_ts=True, max_duration= 20):
    tqdm.write(f"[INFO] Processing: {audio_path}")
    diar_dir   = get_output_dir(audio_path, "diarization")
    ts_path    = os.path.join(diar_dir, "single_timestamps.json")
    rem_dir    = get_output_dir(audio_path, "rematch")
    out_json   = os.path.join(rem_dir, "rematched_timestamps.json")

    ###
    if keep_diar_ts:
        tqdm.write(f"Keep original diarization timestamps!")
    
    else:
        tqdm.write(f"Use only combined timestamps!")
    ###
    
    if skip_if_exist and os.path.isfile(out_json):
        tqdm.write(f"[SKIP] {out_json} exists")
        return

    if not os.path.isfile(ts_path):
        tqdm.write(f"[WARN] Missing timestamp file → {ts_path}")
        return

    original_ts = load_json(ts_path)
    combined_ts = combine_same_speaker(original_ts, max_duration=max_duration)

    chunks, filt_ts = separate_audio(audio_path, combined_ts)
    if not chunks:
        tqdm.write(f"[WARN] No valid chunks for {audio_path}")
        return

    if len(chunks) < 5:
        msg = f"[SKIP] Too few chunks ({len(chunks)}) for {audio_path}, skipping sample"
        tqdm.write(msg)
        failed_rematch_list.append(f"{audio_path} [too few chunks]")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(
        source=DEFAULT_MODEL, run_opts={"device": device},
        savedir="pretrained_models/spkrec"
    )

    embeds = encode_audio_chunks(chunks, classifier)
    labels, _, _ = embeddings_clustering(embeds, n_components=n_components)
    if labels is None:
        tqdm.write(f"[SKIP] Skipping {audio_path} due to failed dimensionality reduction")
        return
    labeled_combined = assign_labels(filt_ts, labels)

    os.makedirs(rem_dir, exist_ok=True)
    write_json(labeled_combined, out_json)
    tqdm.write(f"[DONE] {out_json}")

    if keep_diar_ts:
        from collections import defaultdict

        def get_longest_segment_per_speaker(timestamps):
            speaker_segments = defaultdict(list)
            for seg in timestamps:
                speaker_segments[seg["speaker"]].append(seg)

            longest = {}
            for spk, segs in speaker_segments.items():
                longest[spk] = max(segs, key=lambda x: x["end"] - x["start"])
            return longest

        def find_overlap_global_speaker(local_seg, global_segs):
            overlaps = []
            for gs in global_segs:
                latest_start = max(local_seg["start"], gs["start"])
                earliest_end = min(local_seg["end"], gs["end"])
                overlap = max(0.0, earliest_end - latest_start)
                if overlap > 0:
                    overlaps.append((overlap, gs["speaker"]))
            if not overlaps:
                return None
            return max(overlaps, key=lambda x: x[0])[1]

        def assign_global_speaker_ids(original_ts, labeled_combined):
            mapping = {}
            longest_per_spk = get_longest_segment_per_speaker(original_ts)
            for local_spk, local_seg in longest_per_spk.items():
                global_spk = find_overlap_global_speaker(local_seg, labeled_combined)
                if global_spk is not None:
                    mapping[local_spk] = global_spk

            final = []
            for seg in original_ts:
                local_spk = seg["speaker"]
                if local_spk not in mapping:
                    continue
                seg["speaker"] = mapping[local_spk]
                final.append(seg)
            return final, mapping

        diar_rematch_json = os.path.join(rem_dir, "rematched_diarization.json")
        rematched_ts, speaker_mapping = assign_global_speaker_ids(original_ts, labeled_combined)
        write_json(rematched_ts, diar_rematch_json)
        tqdm.write(f"[DONE] {diar_rematch_json}")

        # Save mapping as separate file
        mapping_json = os.path.join(rem_dir, "speaker_mapping.json")
        write_json(speaker_mapping, mapping_json)
        tqdm.write(f"[DONE] {mapping_json}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rematch/cluster speaker segments for every audio file in a folder"
    )
    parser.add_argument("--input_folder_path", required=True,
                        help="Folder containing audio files")
    parser.add_argument("--skip_if_exist", action="store_true",
                        help="Skip if rematched_timestamps.json already exists")
    parser.add_argument("--n_components", type=int, default=10,
                        help="UMAP embedding dimensions")
    parser.add_argument(
        "--no_keep_diar_ts",
        dest="keep_diar_ts",
        action="store_false",
        help="Disable keeping diarization timestamps (default is True)"
    )
    parser.add_argument("--max_duration", type=float, default=20.0,
                        help="Maximum duration in seconds for a combined audio segment (default: 20.0)")
    args = parser.parse_args()

    audio_paths = get_audio_paths(args.input_folder_path)
    print(f"[INFO] {len(audio_paths)} audio files detected")

    failed_rematch_list = []  # to show the whole list of failure sample names

    for ap in tqdm(audio_paths, desc="Rematching", unit="file"):
        try:
            process_one_audio(
                ap,
                skip_if_exist=args.skip_if_exist,
                n_components=args.n_components,
                keep_diar_ts=args.keep_diar_ts,
                max_duration=args.max_duration
            )
        except Exception as e:
            import traceback
            err_msg = f"[ERROR] Failed processing: {ap}\n{traceback.format_exc()}"
            print(err_msg, file=sys.stderr)
            tqdm.write(err_msg)
            failed_rematch_list.append(ap)  # add the failure sample name

    
    if failed_rematch_list:
        print(f"\n[SUMMARY] {len(failed_rematch_list)} samples failed during rematch:")
        for fail in failed_rematch_list:
            print(f" - {fail}")
    else:
        print("\n[SUMMARY] All samples processed successfully without rematch failure.")

