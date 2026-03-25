from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .constants import MAX_CHARS_PER_SENT, MAX_SENTENCES, ZONE_ORDER
from .io import get_text, passthrough_fields
from .model import PointerModel
from .text_utils import build_sentence_slices, find_z1_anchor, find_z4_anchor
from .vocab import load_vocab


def _pick_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def downsample_sentences(slices: List[Tuple[int, int]], max_sentences: int) -> List[Tuple[int, int]]:
    if len(slices) <= max_sentences:
        return slices
    step = max(1, len(slices) // max_sentences)
    sampled = slices[::step]
    if sampled and sampled[-1][1] != slices[-1][1]:
        sampled.append(slices[-1])
    return sampled[:max_sentences]


def encode_sentence(text: str, vocab: Dict[str, int], unk_id: int) -> List[int]:
    ids: List[int] = []
    for ch in text:
        if not ch.strip():
            continue
        ids.append(vocab.get(ch, unk_id))
        if len(ids) >= MAX_CHARS_PER_SENT:
            break
    if not ids:
        ids = [unk_id]
    return ids


def monotonic_fix(boundaries: List[int], text_len: int) -> List[int]:
    fixed: List[int] = []
    last = 0
    for b in boundaries:
        b = max(last, int(b))
        b = min(text_len, b)
        fixed.append(b)
        last = b
    return fixed


def enforce_anchor_auto(boundaries: List[int], text: str) -> Tuple[List[int], Dict, str]:
    """
    Anchor policy: auto
    - If anchors exist and are valid, enforce them.
    - Otherwise, keep boundaries unchanged but report status.
    """
    z1_end = find_z1_anchor(text)
    z4_end = find_z4_anchor(text)
    anchor_end = {"Z1_Header": z1_end, "Z4_Reasoning": z4_end}

    if z1_end is None or z4_end is None:
        return boundaries, anchor_end, "missing_anchor"
    if not (0 < z1_end < z4_end <= len(text)):
        return boundaries, anchor_end, "invalid_anchor_order"

    anchored = list(boundaries)
    if anchored:
        anchored[0] = z1_end
    if len(anchored) > 3:
        anchored[3] = z4_end

    anchored = [min(max(0, int(b)), len(text)) for b in anchored]

    # Ensure left side does not exceed the Z4 anchor.
    if len(anchored) > 3:
        for i in range(2, -1, -1):
            anchored[i] = min(anchored[i], anchored[i + 1])

    # Enforce monotonicity left-to-right.
    for i in range(1, len(anchored)):
        if anchored[i] < anchored[i - 1]:
            anchored[i] = anchored[i - 1]

    # Re-apply anchors and fix right side monotonicity.
    anchored[0] = z1_end
    if len(anchored) > 3:
        anchored[3] = z4_end
        for i in range(4, len(anchored)):
            if anchored[i] < anchored[i - 1]:
                anchored[i] = anchored[i - 1]

    return anchored, anchor_end, "ok"


@dataclass
class Predictor:
    model_path: Optional[Path] = None
    vocab_path: Optional[Path] = None
    device: str = "cuda"
    anchor: str = "auto"

    def __post_init__(self) -> None:
        bundle_root = Path(__file__).resolve().parents[2]
        assets = bundle_root / "assets"
        self.model_path = Path(self.model_path) if self.model_path else assets / "best_model.pt"
        self.vocab_path = Path(self.vocab_path) if self.vocab_path else assets / "vocab.json"

        self._device = _pick_device(self.device)
        self._vocab = load_vocab(self.vocab_path)
        self._pad_id = self._vocab["<pad>"]
        self._unk_id = self._vocab["<unk>"]

        self._model = PointerModel(vocab_size=len(self._vocab), pad_id=self._pad_id)
        state = torch.load(self.model_path, map_location=self._device)
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()

    @property
    def torch_device(self) -> torch.device:
        return self._device

    def _build_batch(self, text: str):
        slices = downsample_sentences(build_sentence_slices(text), MAX_SENTENCES)
        if not slices:
            return None
        boundary_positions = [s[1] for s in slices]
        max_sents = len(slices)

        sent_chars = torch.full(
            (1, max_sents, MAX_CHARS_PER_SENT),
            self._pad_id,
            dtype=torch.long,
        )
        sent_mask = torch.zeros((1, max_sents), dtype=torch.bool)
        boundary_pos = torch.tensor(boundary_positions, dtype=torch.long).unsqueeze(0)

        for j, (start, end) in enumerate(slices):
            ids = encode_sentence(text[start:end], self._vocab, self._unk_id)
            sent_chars[0, j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            sent_mask[0, j] = True

        return {
            "sent_chars": sent_chars,
            "sent_mask": sent_mask,
            "boundary_positions": boundary_pos,
            "boundary_positions_list": boundary_positions,
        }

    def predict_text(self, text: str, extra_fields: Optional[Dict] = None) -> Dict:
        extra_fields = extra_fields or {}
        text = text or ""
        batch = self._build_batch(text)
        if batch is None:
            out = {
                **extra_fields,
                "text_length": len(text),
                "anchor_end": {"Z1_Header": None, "Z4_Reasoning": None},
                "anchor_status": "empty_or_unsplittable",
                "boundaries": [],
                "zones": {z: {"start": 0, "end": 0, "text": ""} for z in ZONE_ORDER},
            }
            return out

        sent_chars = batch["sent_chars"].to(self._device)
        sent_mask = batch["sent_mask"].to(self._device)
        boundary_positions = batch["boundary_positions"].to(self._device)

        with torch.no_grad():
            logits, _ = self._model(sent_chars, sent_mask)
            pred_idx = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()

        boundary_pos = boundary_positions.squeeze(0).cpu().tolist()
        boundaries = [boundary_pos[i] for i in pred_idx]
        boundaries = monotonic_fix(boundaries, len(text))

        if self.anchor == "auto":
            boundaries, anchor_end, anchor_status = enforce_anchor_auto(boundaries, text)
        elif self.anchor == "off":
            anchor_end = {"Z1_Header": find_z1_anchor(text), "Z4_Reasoning": find_z4_anchor(text)}
            anchor_status = "off"
        else:
            raise ValueError(f"Unsupported anchor policy: {self.anchor}")

        zones: Dict[str, Dict] = {}
        for idx, zone in enumerate(ZONE_ORDER):
            if idx < len(boundaries):
                start = 0 if idx == 0 else boundaries[idx - 1]
                end = boundaries[idx]
            else:
                start = boundaries[-1] if boundaries else 0
                end = len(text)
            zones[zone] = {"start": start, "end": end, "text": text[start:end]}

        return {
            **extra_fields,
            "text_length": len(text),
            "anchor_end": anchor_end,
            "anchor_status": anchor_status,
            "boundaries": boundaries,
            "zones": zones,
        }

    def predict_record(self, record: Dict) -> Dict:
        text = get_text(record)
        extra = passthrough_fields(record)
        return self.predict_text(text, extra_fields=extra)


def partition_text(
    text: str,
    model_path: Optional[Path] = None,
    vocab_path: Optional[Path] = None,
    device: str = "cuda",
    anchor: str = "auto",
) -> Dict:
    pred = Predictor(model_path=model_path, vocab_path=vocab_path, device=device, anchor=anchor)
    return pred.predict_text(text)


def write_run_meta(path: Path, meta: Dict) -> None:
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def default_run_dir(output_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / ts

