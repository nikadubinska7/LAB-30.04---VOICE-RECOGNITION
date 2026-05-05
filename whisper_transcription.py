#LAB | Whisper STT Implementation
#Author: Nika Dubynska

import os
import re
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment


# --------------------------------------------------
# Environment setup
# --------------------------------------------------

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio"
CHUNKS_DIR = BASE_DIR / "chunks"
OUTPUTS_DIR = BASE_DIR / "outputs"

AUDIO_DIR.mkdir(exist_ok=True)
CHUNKS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Guided transcription prompt
# --------------------------------------------------

HAWAII_GUIDED_PROMPT = """
The audio is an interview about hukilau, a Hawaiian fishing method.

Expected vocabulary:
hukilau, lau, leaf, floaters, dried coconut, rope, net, top of the net,
bottom of the net, pocket on the net, lead sinkers, two boats, shore,
forty or fifty guys, school of fish, akule, aji.

Context:
The speakers discuss how a hukilau net is carried out by boats, pulled in by
people on shore, and uses leaves near the top to scare fish so they do not jump
over the rope. The bottom of the net has lead, not lid. The net has a pocket
where fish get caught. Akule is a fish; aji is the Japanese name.

Instructions:
Transcribe only audible speech.
Do not add speaker labels.
Do not add explanations.
Do not invent missing words.
Keep hesitations, repetitions, and informal speech.
"""


# --------------------------------------------------
# Audio verification
# --------------------------------------------------

def check_audio_file(audio_path):
    """
    Loads an audio file and prints basic information about it.
    This confirms that the file exists and can be processed.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = AudioSegment.from_file(audio_path)
    duration_seconds = len(audio) / 1000

    print("\nAudio file loaded successfully.")
    print(f"File name: {audio_path.name}")
    print(f"Duration: {duration_seconds:.2f} seconds")
    print(f"Channels: {audio.channels}")
    print(f"Frame rate: {audio.frame_rate} Hz")
    print(f"Sample width: {audio.sample_width} bytes")

    return audio


# --------------------------------------------------
# Audio chunking
# --------------------------------------------------

def split_audio_into_chunks(audio_path, chunk_length_minutes=1):
    """
    Splits an audio file into smaller chunks.

    Each chunk is exported as an MP3 file into the chunks/ directory.
    The function returns a list of dictionaries containing:
    - chunk path
    - chunk index
    - start time in milliseconds
    - end time in milliseconds
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = AudioSegment.from_file(audio_path)

    chunk_length_ms = chunk_length_minutes * 60 * 1000
    total_length_ms = len(audio)

    chunks = []

    for old_chunk in CHUNKS_DIR.glob(f"{audio_path.stem}_chunk_*.mp3"):
        old_chunk.unlink()

    print(f"\nSplitting audio into {chunk_length_minutes}-minute chunks...")

    for start_ms in range(0, total_length_ms, chunk_length_ms):
        end_ms = min(start_ms + chunk_length_ms, total_length_ms)
        chunk_audio = audio[start_ms:end_ms]

        chunk_index = len(chunks) + 1
        chunk_filename = f"{audio_path.stem}_chunk_{chunk_index:03d}.mp3"
        chunk_path = CHUNKS_DIR / chunk_filename

        chunk_audio.export(chunk_path, format="mp3")

        chunks.append({
            "index": chunk_index,
            "path": chunk_path,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms
        })

        print(
            f"Created chunk {chunk_index}: "
            f"{chunk_path.name} "
            f"({start_ms / 1000:.2f}s - {end_ms / 1000:.2f}s)"
        )

    print(f"\nTotal chunks created: {len(chunks)}")

    return chunks


# --------------------------------------------------
# Timestamp helpers
# --------------------------------------------------

def format_timestamp(seconds):
    """
    Converts seconds into HH:MM:SS.mmm format.
    Example: 65.25 -> 00:01:05.250
    """
    milliseconds = int((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def format_srt_timestamp(seconds):
    """
    Converts seconds into SRT timestamp format.
    Example: 65.25 -> 00:01:05,250
    """
    milliseconds = int((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def extract_adjusted_segments(transcription, chunk_start_ms):
    """
    Extracts timestamped segments from a chunk transcription
    and adjusts each segment to match the original full-audio timeline.
    """
    adjusted_segments = []
    chunk_offset_seconds = chunk_start_ms / 1000

    segments = getattr(transcription, "segments", [])

    for segment in segments:
        adjusted_start = segment.start + chunk_offset_seconds
        adjusted_end = segment.end + chunk_offset_seconds

        adjusted_segments.append({
            "start": adjusted_start,
            "end": adjusted_end,
            "start_formatted": format_timestamp(adjusted_start),
            "end_formatted": format_timestamp(adjusted_end),
            "text": segment.text.strip()
        })

    return adjusted_segments


# --------------------------------------------------
# Transcription functions
# --------------------------------------------------

def transcribe_basic(audio_path):
    """
    Transcribes an audio file without chunking and without a prompt.
    This is the unguided baseline transcription.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"\nStarting basic transcription for: {audio_path.name}")

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    print("\nBasic transcription completed.")
    print("\nBasic transcription text:\n")
    print(transcription.text)

    return transcription


def transcribe_with_prompt(audio_path, prompt):
    """
    Transcribes an audio file using a guiding prompt.
    The prompt gives Whisper context about vocabulary, names, and expected spelling.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"\nStarting guided transcription for: {audio_path.name}")

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            prompt=prompt
        )

    print("\nGuided transcription completed.")
    print("\nGuided transcription text:\n")
    print(transcription.text)

    return transcription


def transcribe_chunks_with_timestamps(chunks, prompt=None):
    """
    Transcribes each chunk and combines all segment-level timestamps
    into one full transcript timeline.

    If prompt is provided, the chunk transcription is guided.
    If prompt is None, the chunk transcription is unguided.
    """
    all_segments = []

    print("\nStarting chunked transcription...")

    for chunk in chunks:
        chunk_path = chunk["path"]
        chunk_start_ms = chunk["start_ms"]

        print(f"\nTranscribing chunk {chunk['index']}: {chunk_path.name}")

        with open(chunk_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                prompt=prompt
            )

        adjusted_segments = extract_adjusted_segments(
            transcription=transcription,
            chunk_start_ms=chunk_start_ms
        )

        all_segments.extend(adjusted_segments)

        print(f"Added {len(adjusted_segments)} timestamped segments.")

    print("\nChunked transcription completed.")
    print(f"Total timestamped segments: {len(all_segments)}")

    return all_segments


# --------------------------------------------------
# Save and load helpers
# --------------------------------------------------

def save_transcription_text(transcription, output_filename):
    """
    Saves transcription text to the outputs folder.
    """
    output_path = OUTPUTS_DIR / output_filename

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(transcription.text)

    print(f"\nSaved transcription to: {output_path}")


def save_timestamped_txt(segments, output_filename):
    """
    Saves timestamped segments as a readable text transcript.
    """
    output_path = OUTPUTS_DIR / output_filename

    with open(output_path, "w", encoding="utf-8") as file:
        for segment in segments:
            file.write(
                f"[{segment['start_formatted']} - {segment['end_formatted']}] "
                f"{segment['text']}\n"
            )

    print(f"\nSaved timestamped transcript to: {output_path}")


def save_timestamped_json(segments, output_filename):
    """
    Saves timestamped transcription segments as JSON.
    JSON is useful for search, filtering, dashboards, or later analysis.
    """
    output_path = OUTPUTS_DIR / output_filename

    export_data = {
        "segment_count": len(segments),
        "segments": segments
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(export_data, file, indent=2, ensure_ascii=False)

    print(f"\nSaved JSON transcript to: {output_path}")


def save_timestamped_srt(segments, output_filename):
    """
    Saves timestamped transcription segments in SRT subtitle format.
    SRT files can be used with video players and captioning tools.
    """
    output_path = OUTPUTS_DIR / output_filename

    with open(output_path, "w", encoding="utf-8") as file:
        for index, segment in enumerate(segments, start=1):
            start = format_srt_timestamp(segment["start"])
            end = format_srt_timestamp(segment["end"])
            text = segment["text"]

            file.write(f"{index}\n")
            file.write(f"{start} --> {end}\n")
            file.write(f"{text}\n\n")

    print(f"\nSaved SRT transcript to: {output_path}")


def export_all_timestamped_formats(segments, base_filename):
    """
    Exports timestamped transcription in TXT, JSON, and SRT formats.
    """
    save_timestamped_txt(
        segments=segments,
        output_filename=f"{base_filename}.txt"
    )

    save_timestamped_json(
        segments=segments,
        output_filename=f"{base_filename}.json"
    )

    save_timestamped_srt(
        segments=segments,
        output_filename=f"{base_filename}.srt"
    )


def read_text_file(file_path):
    """
    Reads a UTF-8 text file and returns its content.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def clean_reference_text(reference_text):
    """
    Removes speaker labels and transcript-only context markers from the reference text.

    The original reference contains information that is not spoken in the audio,
    such as CASSIDY:, INFORMANT:, SECOND INFORMANT:, brackets, and braces.
    Those should not be treated as Whisper transcription errors.
    """
    text = reference_text

    text = re.sub(r"(?m)^\s*(CASSIDY|INFORMANT|SECOND INFORMANT):\s*", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# LLM comparison
# --------------------------------------------------

def compare_transcriptions_with_llm(reference_text, basic_text, guided_text):
    """
    Uses an OpenAI language model to compare unguided and guided transcriptions
    against the cleaned original reference transcript.
    """
    print("\nStarting LLM comparison analysis...")

    comparison_prompt = f"""
You are evaluating speech-to-text transcription quality.

The reference transcript has already been cleaned to remove non-audible metadata
such as speaker labels, bracketed notes, and transcript-only context.

Compare two Whisper transcriptions against the cleaned reference:

1. BASIC_TRANSCRIPTION: produced without a prompt.
2. GUIDED_TRANSCRIPTION: produced with a vocabulary/context prompt.

Focus only on audible content:
- words and phrases
- Hawaiian terms
- meaning preservation
- missing content
- inserted content
- punctuation only when it affects readability or meaning

Do not penalize either transcription for missing speaker names, speaker labels,
bracket notes, or other information that would not be heard in the audio.

Return a concise report with these sections:
1. Overall winner
2. Accuracy summary
3. Hawaiian vocabulary accuracy
4. Missing or incorrect content
5. Inserted or hallucinated content
6. Practical recommendation
7. Approximate score out of 10 for each transcription

CLEANED_REFERENCE:
{reference_text}

BASIC_TRANSCRIPTION:
{basic_text}

GUIDED_TRANSCRIPTION:
{guided_text}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=comparison_prompt
    )

    print("\nLLM comparison completed.")
    print("\nComparison report:\n")
    print(response.output_text)

    return response.output_text


def save_comparison_report(report_text, output_filename="comparison_report.txt"):
    """
    Saves the LLM comparison report to the outputs folder.
    """
    output_path = OUTPUTS_DIR / output_filename

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    print(f"\nSaved comparison report to: {output_path}")


# --------------------------------------------------
# Main execution for Step 8
# --------------------------------------------------

if __name__ == "__main__":
    print("OpenAI client initialized successfully.")
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"Chunks directory: {CHUNKS_DIR}")
    print(f"Outputs directory: {OUTPUTS_DIR}")

    sample_audio = AUDIO_DIR / "Hawaii.mp3"

    check_audio_file(sample_audio)

    chunks = split_audio_into_chunks(sample_audio, chunk_length_minutes=1)

    timestamped_segments = transcribe_chunks_with_timestamps(
        chunks=chunks,
        prompt=HAWAII_GUIDED_PROMPT
    )

    export_all_timestamped_formats(
        segments=timestamped_segments,
        base_filename="chunked_timestamped_transcription"
    )

    print("\nPreview of timestamped transcription:")
    for segment in timestamped_segments[:5]:
        print(
            f"[{segment['start_formatted']} - {segment['end_formatted']}] "
            f"{segment['text']}"
        )