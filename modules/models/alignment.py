"""
Example:
    python forced_alignment.py \
        --input_folder_path /mmfs1/ytb_aae_data/black_spin_global_2 \
        --skip_if_exist \                # <- add this to skip finished files
        --language en \
        --batch_size 16
"""

import os, json, argparse, torch
from utils import get_audio_paths, get_output_dir
from ctc_forced_aligner import (
    load_audio, load_alignment_model, generate_emissions,
    preprocess_text, get_alignments, get_spans, postprocess_results,
)


def forced_alignment_batch(
    input_folder_path: str,
    language: str = "eng",
    batch_size: int = 16,
    skip_if_exist: bool = False,
):
    """Generate alignment.json for each audio file in *input_folder_path*."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"[MODEL] Loading alignment model on {device} (dtype={dtype})")

    model, tokenizer = load_alignment_model(device, dtype=dtype)
    audio_paths = get_audio_paths(input_folder_path)
    print(f"[INFO] {len(audio_paths)} audio files found")

    for audio_path in audio_paths:
        base = os.path.splitext(os.path.basename(audio_path))[0]

        # transcripts
        folder = os.path.dirname(audio_path)
        t1 = os.path.join(folder, f"{base}.en-orig_processed.txt")
        t2 = os.path.join(folder, f"{base}.en_processed.txt")
        if   os.path.exists(t1): transcript = t1
        elif os.path.exists(t2): transcript = t2
        else:
            print(f"[WARN] No transcript for {base} – skipping")
            continue

        # outputs
        out_dir  = get_output_dir(audio_path, "alignment")
        out_json = f"{out_dir}/alignment.json"

        if skip_if_exist and os.path.isfile(out_json):
            print(f"[SKIP] {out_json} already exists")
            continue

        os.makedirs(out_dir, exist_ok=True)
        print(f"[ALIGN] {base}")

        try:
            wav = load_audio(audio_path, model.dtype, model.device)
            text = open(transcript, encoding="utf-8").read().replace("\n", " ").strip()

            emissions, stride = generate_emissions(model, wav, batch_size)
            tokens_star, text_star = preprocess_text(text, romanize=True, language=language)
            segments, scores, blank = get_alignments(emissions, tokens_star, tokenizer)
            spans  = get_spans(tokens_star, segments, blank)
            result = postprocess_results(text_star, spans, stride, scores)

            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[DONE] → {out_json}")

        except Exception as e:
            print(f"[ERROR] {audio_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder_path", required=True,
                        help="Folder that contains audio files")
    parser.add_argument("--language", default="eng",
                        help="Language code for preprocess_text()")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Emission-generation batch size")
    parser.add_argument("--skip_if_exist", action="store_true",
                        help="Skip alignment when alignment.json already exists")
    args = parser.parse_args()

    forced_alignment_batch(
        input_folder_path=args.input_folder_path,
        language=args.language,
        batch_size=args.batch_size,
        skip_if_exist=args.skip_if_exist,
    )