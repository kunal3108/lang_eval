"""Run 3 — LLM reconciliation of the full-pass and chunked transcripts.

Prompt modes:
  anchored   — one KEEP/FIX line per chunk (default); the rest come from the notebook
  direct     — return only the final corrected transcript
  two_stage  — CHUNK_FLAGS (CLEAN / GIBBERISH) followed by FINAL_TRANSCRIPT
  segment    — segment-level reconciliation, transcript only

Usage:
  llm_judge.py <full_file> <chunked_file> <model_repo> <max_tokens> <mode> [segments_json]
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit  # noqa: E402


CHUNK_LINE = re.compile(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*\|\s*(\S+)\s*\|\s*(.*)$")
DECISION_LINE = re.compile(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s*\|\s*(KEEP|FIX)\b", re.IGNORECASE)
SENTENCE_SPLIT = re.compile(r"(?<=[.?!।])\s+")


def parse_chunks(chunked_transcript):
    """Turn the run-2 lines back into (start, end, language, text) records."""
    chunks = []
    for line in chunked_transcript.splitlines():
        match = CHUNK_LINE.match(line)
        if match:
            chunks.append({
                "start": float(match.group(1)),
                "end": float(match.group(2)),
                "language": match.group(3),
                "text": match.group(4).strip(),
            })
    return chunks


def full_text_for_chunk(start, end, segments, centred_only=True):
    """The stretch of the full pass covering this chunk's seconds.

    Same idea as the notebook's get_full_text_for_chunk. Judging reads better
    with segments centred in the window; repairing needs every overlapping
    segment, because the sentence a corrupt chunk needs often sits in a segment
    that straddles the boundary.
    """
    overlapping = [
        segment for segment in segments
        if segment["end"] > start and segment["start"] < end
    ]
    if centred_only:
        centred = [
            segment for segment in overlapping
            if start <= (segment["start"] + segment["end"]) / 2 < end
        ]
        overlapping = centred or overlapping
    return " ".join(segment["text"].strip() for segment in overlapping).strip()


def drop_repeated_sentences(text, *neighbours):
    """Full-pass windows overlap, so trim what the neighbouring lines say."""
    already = {
        sentence.strip()
        for neighbour in neighbours if neighbour
        for sentence in SENTENCE_SPLIT.split(neighbour)
    }
    if not already:
        return text
    kept = [
        sentence for sentence in SENTENCE_SPLIT.split(text)
        if sentence.strip() and sentence.strip() not in already
    ]
    return " ".join(kept).strip() or text


def build_blocks(chunks, segments):
    blocks = []
    for chunk in chunks:
        block = (
            f"[{chunk['start']:.1f} - {chunk['end']:.1f}]  language={chunk['language']}\n"
            f"  CHUNK: {chunk['text']}"
        )
        if segments:
            block += f"\n  FULL : {full_text_for_chunk(chunk['start'], chunk['end'], segments)}"
        blocks.append(block)
    return "\n\n".join(blocks)


def assemble(chunks, decisions, segments):
    """Build the final transcript from the model's decisions.

    The model judges; the text itself is copied here, so a KEEP is always the
    chunk verbatim — in its own script — and a FIX is always the aligned stretch
    of the full pass.
    """
    rows = []
    for index, chunk in enumerate(chunks):
        decision = decisions.get(index, "KEEP")
        repair = (
            full_text_for_chunk(chunk["start"], chunk["end"], segments, centred_only=False)
            if segments else ""
        )

        if decision == "FIX" and repair:
            following = chunks[index + 1]["text"] if index + 1 < len(chunks) else ""
            if decisions.get(index + 1) == "FIX":
                following = ""
            text = drop_repeated_sentences(
                repair,
                rows[-1]["text"] if rows else "",
                following,
            )
        else:
            decision = "KEEP"
            text = chunk["text"]

        rows.append({
            "start": chunk["start"],
            "end": chunk["end"],
            "language": chunk["language"],
            "decision": decision,
            "text": text,
        })
    return rows


def parse_decisions(output, chunks):
    """Read the model's KEEP / FIX lines, matched to chunks by timestamp."""
    by_start = {round(chunk["start"], 1): index for index, chunk in enumerate(chunks)}
    decisions = {}
    order = 0

    for line in output.splitlines():
        match = DECISION_LINE.match(line)
        if not match:
            continue
        start = round(float(match.group(1)), 1)
        index = by_start.get(start, order if order < len(chunks) else None)
        if index is not None:
            decisions[index] = match.group(3).upper()
        order += 1

    return decisions


def prompt_anchored(chunks, segments):
    """Ask only for a judgement per chunk — never for the transcript text itself.

    Asking the model to write the transcript lets it drift back to the fluent
    English of the full pass and translate good Hindi away, or blend the two
    readings together. Deciding KEEP / FIX per chunk is the part that needs
    judgement; assemble() does the copying.
    """
    return f"""
You are judging two ASR readings of the SAME Hindi-English (Hinglish) call, chunk by chunk.

CHUNK — Whisper run on that short stretch alone, detecting its language independently.
  The language tag and the script show what was ACTUALLY spoken there.
  Some chunks are corrupted: broken non-words, nonsense repetition, hallucinations.

FULL — the same seconds heard by one Whisper pass over the whole call, FORCED to English.
  Fluent, but any Hindi speech was TRANSLATED into English and it spills into neighbouring
  seconds. English here is NOT evidence that English was spoken.

For each block, answer one question: is the CHUNK text intelligible?

Are its words real words, and do they form something a customer or an agent could
plausibly say on this call?

KEEP — intelligible. Imperfect grammar, filler, a cut-off sentence and small ASR slips
  are all still intelligible. Hindi written in Devanagari that makes sense is
  intelligible and must be kept, even though the FULL line shows English for that
  moment — that English is only a translation of the Hindi that was really spoken.

FIX — corrupt. The words themselves are broken non-words, or nonsense repetition, or the
  text is plainly unrelated to this conversation.

When in doubt, answer KEEP. Only clear corruption deserves FIX.

Judge each block on its own, and do not let one bad chunk make you distrust the rest.

OUTPUT FORMAT

One line per block, in order, carrying that block's start and end and nothing else:

<start> - <end> | KEEP
<start> - <end> | FIX

There are {len(chunks)} blocks, so output exactly {len(chunks)} lines.
Never write any transcript text — only these decisions. No commentary, no blank lines.

EXAMPLE

[15.0 - 20.0]  language=hi
  CHUNK: अपके अवाज थोड़ी साप नहीं आरी है।
  FULL : Am I speaking to Mr.Mohammad? Yes, yes. Your voice is not clear.

[25.0 - 30.0]  language=hi
  CHUNK: या आइँ आप आटिकल नहीं नहीं। या थाइंकि या पर पर कणफरमिंग महामर।
  FULL : Yeah, you are audible now. Thank you so much for confirming Mohammed.

Correct answer:
15.0 - 20.0 | KEEP
25.0 - 30.0 | FIX

The first is real Hindi that makes sense, so it is kept even though FULL shows an English
translation. The second is broken non-words ("आइँ", "आटिकल", "थाइंकि", "कणफरमिंग महामर")
with meaningless repetition, so it is fixed.

BLOCKS TO JUDGE:

{build_blocks(chunks, segments)}

DECISIONS:
"""


def prompt_direct(full_transcript, chunked_transcript):
    return f"""
You are a multilingual ASR transcript reconciliation system.

You are given TWO transcripts of the SAME audio.

FULL TRANSCRIPT:
- Generated by running Whisper once on the complete audio.
- Usually has better sentence continuity and context.
- However, it may incorrectly interpret Hindi speech as English because English is the dominant language.

CHUNKED TRANSCRIPT:
- Generated by running Whisper independently on short chunks.
- It is better at detecting Hindi-English language switches.
- However, individual chunks may contain hallucinations, repeated words, broken sentences, or nonsense.

Your job is to produce ONE final corrected transcript.

RULES:

1. Compare both transcripts using surrounding context.
2. Preserve genuine Hindi-English switching.
3. If the chunked transcript contains meaningful Hindi that fits the conversation, prefer it over an English mistranscription in the full transcript.
4. If a chunk contains nonsense or hallucinated text, prefer the corresponding meaningful sentence from the full transcript.
5. Examples of hallucinations include:
   - unrelated words
   - excessive repetition
   - impossible sentences
   - random names or phrases
6. Do not invent information that appears in neither transcript.
7. Do not translate Hindi into English.
8. Do not translate English into Hindi.
9. Preserve the chronological order of the conversation.
10. Make only corrections needed to reconcile the two ASR outputs.
11. Prefer semantic coherence with the surrounding conversation.
12. Output ONLY the final corrected transcript.

FULL_TRANSCRIPT:
<<<
{full_transcript}
>>>

CHUNKED_TRANSCRIPT:
<<<
{chunked_transcript}
>>>

FINAL CORRECTED TRANSCRIPT:
"""


def prompt_segment(full_transcript, chunked_transcript):
    return f"""
You are reconciling two ASR transcripts of the SAME audio segment.

FULL_TRANSCRIPT:
{full_transcript}

CHUNKED_TRANSCRIPT:
{chunked_transcript}

The two transcripts have different characteristics:

FULL_TRANSCRIPT:
- Usually preserves meaning and sentence continuity well.
- It was forced to English.
- Therefore Hindi speech may have been TRANSLATED into English.
- Do NOT assume English text means English was actually spoken.

CHUNKED_TRANSCRIPT:
- Generated with automatic language detection.
- Usually preserves the language actually spoken.
- However, short chunks may contain hallucinations, repetition,
  broken words, or meaningless text.

Your task is to return the transcript that most faithfully represents
WHAT WAS ACTUALLY SPOKEN.

Follow these rules in order:

1. LANGUAGE PRESERVATION IS IMPORTANT.

If the chunked transcript is meaningful and its meaning agrees with
the full transcript, prefer the CHUNKED transcript.

Example:

FULL:
"Your voice is not clear."

CHUNKED:
"आपकी आवाज थोड़ी साफ नहीं आ रही है।"

FINAL:
"आपकी आवाज थोड़ी साफ नहीं आ रही है।"

The English sentence is likely a translation of the Hindi speech,
so preserve the Hindi.

2. DO NOT TRANSLATE.

Never convert meaningful Hindi into English merely because the full
transcript is English.

Never convert meaningful English into Hindi.

Preserve the language actually represented by the reliable chunked
transcript.

3. DETECT BAD CHUNK TRANSCRIPTIONS.

Consider the chunked transcript unreliable if it contains:
- meaningless word sequences
- obvious hallucinations
- excessive repetition
- unrelated words
- broken grammar that has no interpretable meaning
- text inconsistent with surrounding conversation

4. IF THE CHUNKED TRANSCRIPT IS CLEARLY BAD:

Use the corresponding FULL_TRANSCRIPT text instead.

Example:

FULL:
"Yeah, you are audible now. Thank you so much for confirming Mohammed."

CHUNKED:
"या आइँ आप आटिकल नहीं नहीं। या थाइंकि या पर पर कणफरमिंग महामर।"

FINAL:
"Yeah, you are audible now. Thank you so much for confirming Mohammed."

5. If both versions are meaningful but differ slightly,
choose the version that:
- preserves the original spoken language
- preserves meaning
- fits the conversational context

6. Do not invent words that occur in neither transcript.

7. Do not paraphrase unnecessarily.

Return ONLY the final corrected transcript for this segment.
"""


def prompt_two_stage(full_transcript, chunked_transcript):
    return f"""
You are a multilingual ASR transcript reconciliation system.

You are given TWO transcripts of the SAME audio.

FULL_TRANSCRIPT:
- Generated by running Whisper once on the complete audio.
- Usually has better sentence continuity and semantic coherence.
- However, it was forced to English.
- Therefore Hindi speech may sometimes appear translated into English.

CHUNKED_TRANSCRIPT:
- Generated by running Whisper independently on short audio chunks.
- Each chunk contains a timestamp, detected language, and transcript.
- It is generally better at preserving Hindi-English language switches.
- However, some chunks may contain hallucinations, repetition, broken words, or meaningless text.

Your task has TWO stages.

STAGE 1 — CLASSIFY EACH CHUNK

For every chunk in CHUNKED_TRANSCRIPT, assign exactly one flag:

CLEAN
or
GIBBERISH

Use CLEAN when:
- the text has understandable meaning
- it forms a plausible statement in Hindi, English, or Hinglish
- minor spelling or ASR errors are acceptable
- grammar may be imperfect but meaning is still understandable

Use GIBBERISH when:
- the text has no coherent meaning
- it contains severe hallucinations
- excessive meaningless repetition
- unrelated random words
- badly corrupted words that make the sentence uninterpretable
- the output clearly does not represent meaningful speech

IMPORTANT:
Do NOT mark meaningful Hindi as gibberish merely because the FULL transcript contains an English translation of it.

Example:

FULL:
"Your voice is not clear."

CHUNK:
[15-20] hi | "आपके आवाज थोड़ी साफ नहीं आ रही है।"

FLAG:
CLEAN

The Hindi should be preserved because it has coherent meaning.

Example:

FULL:
"Yeah, you are audible now. Thank you so much for confirming Mohammed."

CHUNK:
[25-30] hi | "या आइँ आप आटिकल नहीं नहीं। या थाइंकि या पर पर कणफरमिंग महामर।"

FLAG:
GIBBERISH


STAGE 2 — BUILD THE FINAL TRANSCRIPT

For each chunk:

IF FLAG = CLEAN:
- prefer the CHUNKED transcript
- preserve the language actually present in the chunk
- do NOT translate Hindi into English
- do NOT translate English into Hindi

IF FLAG = GIBBERISH:
- replace that chunk with the corresponding meaningful content from FULL_TRANSCRIPT
- use surrounding context and timestamps to identify the relevant full-transcript sentence
- do not copy unrelated sentences from before or after the corrupted chunk

GENERAL RULES:

1. Preserve chronological order.

2. Preserve genuine Hindi-English switching.

3. Do not invent content that appears in neither transcript.

4. Do not paraphrase unnecessarily.

5. Minor grammatical or spelling errors do NOT automatically make a chunk gibberish.

6. Meaningfulness is more important than perfect grammar.

7. If a Hindi chunk is meaningful and approximately corresponds in meaning to an English sentence in FULL_TRANSCRIPT, keep the Hindi chunk.

8. If the chunked text is unintelligible but FULL_TRANSCRIPT contains a coherent version of that same portion, use the FULL text.

9. Do not reject the entire CHUNKED_TRANSCRIPT because one chunk is gibberish. Evaluate EACH CHUNK independently.

10. Do not output analysis or explanations.

Return your answer in EXACTLY this format:

CHUNK_FLAGS:
[
    {{"start": "00:00:00", "end": "00:00:05", "flag": "CLEAN"}},
    {{"start": "00:00:05", "end": "00:00:10", "flag": "CLEAN"}},
    {{"start": "00:00:25", "end": "00:00:30", "flag": "GIBBERISH"}}
]

FINAL_TRANSCRIPT:
<final corrected transcript only>

FULL_TRANSCRIPT:
<<<
{full_transcript}
>>>

CHUNKED_TRANSCRIPT:
<<<
{chunked_transcript}
>>>
"""


PROMPTS = {
    "direct": prompt_direct,
    "two_stage": prompt_two_stage,
    "segment": prompt_segment,
}


def main():
    full_path = sys.argv[1]
    chunked_path = sys.argv[2]
    model_repo = sys.argv[3]
    max_tokens = int(sys.argv[4])
    mode = sys.argv[5] if len(sys.argv) > 5 else "anchored"
    segments_path = sys.argv[6] if len(sys.argv) > 6 else None

    if mode != "anchored" and mode not in PROMPTS:
        raise ValueError(f"unknown prompt mode '{mode}'")

    with open(full_path, encoding="utf-8") as handle:
        full_transcript = handle.read()

    with open(chunked_path, encoding="utf-8") as handle:
        chunked_transcript = handle.read()

    segments = []
    if segments_path and os.path.exists(segments_path):
        with open(segments_path, encoding="utf-8") as handle:
            segments = json.load(handle)

    started = time.time()
    emit(event="start", msg=f"loading {model_repo.split('/')[-1]}")

    from mlx_lm import load, stream_generate

    model, tokenizer = load(model_repo)

    chunks = parse_chunks(chunked_transcript) if mode == "anchored" else []

    if mode == "anchored":
        if not chunks:
            raise ValueError("no chunk lines to judge")
        prompt = prompt_anchored(chunks, segments)
    else:
        prompt = PROMPTS[mode](full_transcript, chunked_transcript)

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    emit(event="progress", pct=3.0, msg=f"model ready, reconciling ({mode})")

    produced = 0
    pieces = []
    finish_reason = None

    for response in stream_generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
    ):
        pieces.append(response.text)
        produced += 1
        finish_reason = response.finish_reason

        emit(event="delta", text=response.text)

        if produced % 10 == 0:
            emit(
                event="progress",
                pct=min(99.0, 100.0 * produced / max_tokens),
                msg=f"{produced} tokens generated  ·  {response.generation_tps:.1f} tok/s",
            )

    truncated = finish_reason == "length"
    output = "".join(pieces).strip()

    if mode == "anchored":
        # The model only judged; the transcript itself is copied here, so a KEEP
        # is the chunk verbatim and a FIX is the aligned full-pass stretch.
        decisions = parse_decisions(output, chunks)
        rows = assemble(chunks, decisions, segments)
        fixed = sum(1 for row in rows if row["decision"] == "FIX")
        emit(
            event="text",
            text=" ".join(row["text"] for row in rows).strip(),
            rows=rows,
            judged=len(decisions),
            fixed=fixed,
            mode=mode,
            truncated=truncated,
        )
    else:
        emit(event="text", text=output, mode=mode, truncated=truncated)
    emit(
        event="done",
        pct=100.0,
        elapsed=round(time.time() - started, 1),
        msg=(
            f"stopped at the {max_tokens}-token cap · {produced} tokens"
            if truncated
            else (
                f"{len(chunks)} chunks judged · {produced} tokens"
                if mode == "anchored"
                else f"{produced} tokens generated"
            )
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit(event="error", msg=f"{type(exc).__name__}: {exc}")
        sys.exit(1)
