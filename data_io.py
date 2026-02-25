"""Data loading for STAPLE EN->HU files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


GoldPrompt = Dict[str, object]
GoldDict = Dict[str, GoldPrompt]
AwsDict = Dict[str, str]


def _split_blocks(path: str | Path) -> List[List[str]]:
    text = Path(path).read_text(encoding="utf-8")
    blocks: List[List[str]] = []
    cur: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def parse_gold(path: str | Path) -> GoldDict:
    """Parse gold file into:
    gold[prompt_id] = {"english", "translations", "weights"} with normalized weights.
    """
    out: GoldDict = {}
    for block in _split_blocks(path):
        header = block[0]
        if "|" not in header:
            raise ValueError(f"Invalid header line: {header}")
        prompt_id, english = header.split("|", 1)

        translations: List[str] = []
        weights: List[float] = []
        for line in block[1:]:
            if "|" not in line:
                raise ValueError(f"Invalid translation line for {prompt_id}: {line}")
            tr, w = line.rsplit("|", 1)
            translations.append(tr)
            weights.append(float(w))

        if not translations:
            raise ValueError(f"No gold translations for prompt {prompt_id}")

        total = sum(weights)
        if total <= 0:
            raise ValueError(f"Non-positive total weight for prompt {prompt_id}")
        norm_weights = [w / total for w in weights]

        out[prompt_id] = {
            "english": english,
            "translations": translations,
            "weights": norm_weights,
        }
    return out


def parse_aws_baseline(path: str | Path) -> AwsDict:
    """Parse AWS baseline file into aws[prompt_id] = translation_string."""
    out: AwsDict = {}
    for block in _split_blocks(path):
        header = block[0]
        if "|" not in header:
            raise ValueError(f"Invalid header line: {header}")
        prompt_id, _english = header.split("|", 1)
        pred = block[1] if len(block) > 1 else ""
        out[prompt_id] = pred
    return out


def parse_prompts(path: str | Path) -> Dict[str, str]:
    """Parse prompt headers into prompt_id -> english."""
    out: Dict[str, str] = {}
    for block in _split_blocks(path):
        header = block[0]
        if "|" not in header:
            raise ValueError(f"Invalid header line: {header}")
        prompt_id, english = header.split("|", 1)
        out[prompt_id] = english
    return out
