from flask import Flask, request, jsonify
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from summarizer import Summarizer
import re
from datetime import datetime
import torch

app = Flask(__name__)

summarizer = Summarizer()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")


@app.route("/", methods=["POST"])
def process_question():
    data = request.get_json()
    student_question = data.get("question", "")
    course_name = data.get("course", "Signal")  # ✅ default course if not provided

    # ✅ dynamically build folder path
    FOLDER_NAME = f"{course_name}/"
    grouped_sentences_file = FOLDER_NAME + "grouped_sentences.pkl"
    grouped_sent_to_metadata_file = FOLDER_NAME + "grouped_sent_to_metadata.pkl"
    grouped_sentences_embeddings_file = FOLDER_NAME + "grouped-sentences-embeddings.idx"

    # ✅ load course-specific data
    faiss_index = faiss.read_index(grouped_sentences_embeddings_file)

    with open(grouped_sentences_file, "rb") as f:
        grouped_sentences = pickle.load(f)

    with open(grouped_sent_to_metadata_file, "rb") as f:
        grouped_sent_to_metadata = pickle.load(f)

    def parse_ts(ts):
        ts = ts.replace("-->", "->").replace("—>", "->").replace("−>", "->").strip()
        if not ts or "->" not in ts:
            return datetime.min, datetime.min
        start, end = ts.split("->", 1)
        fmt = "%H:%M:%S.%f"
        start = re.sub(r"[^0-9:.]", "", start.strip())
        end = re.sub(r"[^0-9:.]", "", end.strip())
        try:
            return datetime.strptime(start, fmt), datetime.strptime(end, fmt)
        except ValueError:
            return datetime.strptime("00:00:00.000", fmt), datetime.strptime("00:00:00.000", fmt)

    def clean_sentences(sentences):
        cleaned = []
        seen = set()
        for s in sentences:
            s = s.strip()
            while s and s[-1] in ".!?":
                s = s[:-1].strip()
            if s and s not in seen:
                cleaned.append(s)
                seen.add(s)
        return cleaned

    def extract_file_number(fname):
        match = re.search(r"(\d+)", fname)
        return int(match.group(1)) if match else float("inf")

    # ✅ Encode the student question and find nearest results
    question_embedding = model.encode(student_question, prompt_name="query")
    if question_embedding.ndim == 1:
        question_embedding = np.expand_dims(question_embedding, axis=0)

    distances, indices = faiss_index.search(question_embedding, 10)

    related_results = []
    for idx in indices[0]:
        grouped_sent = grouped_sentences[idx]
        meta = grouped_sent_to_metadata.get(grouped_sent, None)

        if meta:
            filename = meta["filename"]
            timestamp_range = meta["timestamp_range"]
            individual_timestamps = meta.get("individual_timestamps", [])
        else:
            filename = "Unknown"
            timestamp_range = "Unknown"
            individual_timestamps = []

        related_results.append(
            (filename, timestamp_range, grouped_sent, individual_timestamps)
        )

    # ✅ Sort and merge results
    related_results_sorted = sorted(
        related_results,
        key=lambda x: (extract_file_number(x[0]), parse_ts(x[1])[0])
    )

    merged_results = []
    for filename, ts_range, text, indiv_ts in related_results_sorted:
        sentences = clean_sentences(text.split('. '))

        if not merged_results:
            merged_results.append([filename, ts_range, '. '.join(sentences) + '.', indiv_ts])
            continue

        prev = merged_results[-1]
        prev_start, prev_end = parse_ts(prev[1])
        curr_start, curr_end = parse_ts(ts_range)

        if filename == prev[0] and curr_start <= prev_end:
            new_start = prev_start.strftime("%H:%M:%S.%f")[:-3]
            new_end = max(prev_end, curr_end).strftime("%H:%M:%S.%f")[:-3]
            prev[1] = f"{new_start} -> {new_end}"

            prev_sentences = clean_sentences(prev[2].split('. '))
            combined_sentences = prev_sentences + sentences
            prev[2] = '. '.join(clean_sentences(combined_sentences)) + '.'
            prev[3].extend(indiv_ts)
        else:
            merged_results.append([filename, ts_range, '. '.join(sentences) + '.', indiv_ts])

    # ✅ Build in-memory data for JSON response
    flat_sentence_map = {}
    for filename, group_ts, group_text, indiv_ts_list in merged_results:
        indiv_sents = [s.strip() for s in re.split(r'\.\s+', group_text) if s.strip()]
        indiv_ts_list = indiv_ts_list or []
        for i, s in enumerate(indiv_sents):
            if s.endswith('.'):
                s = s[:-1].strip()
            ts = indiv_ts_list[i] if i < len(indiv_ts_list) else group_ts
            flat_sentence_map[s] = (filename, ts)

    all_sentences = list(flat_sentence_map.keys())
    full_text = ". ".join(all_sentences)

    # ✅ Summarization → directly in-memory JSON lists
    def summarize_to_json(ratio):
        summary_text = summarizer(full_text, ratio=ratio)
        summary_sentences = [s.strip() for s in summary_text.split('. ') if s.strip()]
        summary_data = []

        for s in summary_sentences:
            matched_file, matched_ts = "Unknown", "Unknown"
            if s in flat_sentence_map:
                matched_file, matched_ts = flat_sentence_map[s]
            else:
                for orig_sentence, (fname, ts) in flat_sentence_map.items():
                    if s[:25].lower() in orig_sentence.lower() or orig_sentence.lower() in s.lower():
                        matched_file, matched_ts = fname, ts
                        break
            summary_data.append({
                "filename": matched_file,
                "timestamp": matched_ts,
                "sentence": s
            })

        return summary_data

    result = {
        "short_answer": summarize_to_json(0.3),
        "medium_answer": summarize_to_json(0.7),
        "long_answer": [
            {"filename": f, "timestamp": t, "sentence": s}
            for f, t, s, _ in merged_results
        ]
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
