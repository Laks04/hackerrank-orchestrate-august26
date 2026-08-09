"""Multimodal media understanding for image posters/screenshots and voice notes.

Design goal: never crash, always return *something* useful, and use the best
tool actually available in the running environment:

Images
  1. Claude vision (if ANTHROPIC_API_KEY is set and the ``anthropic`` package
     is installed) - asks the model to OCR any visible text and classify the
     visual content (poster, screenshot, invoice, ID/QR, scam-style urgency
     graphic, product photo, etc).
  2. Local OCR via ``pytesseract`` + Pillow, if both the Python packages and
     the system ``tesseract`` binary are available.
  3. Metadata-only fallback: we still know the media exists, its type, and
     can fall back to the surrounding message_text/sender/history signals.

Voice notes
  1. Local ASR via ``speech_recognition`` + ``pydub`` (uses the free Google
     Web Speech endpoint - no API key required, only outbound network).
  2. Metadata-only fallback (file present, best-effort duration via
     ``mutagen`` if installed, else just file size).

Every analyzer result is cached in-process (and optionally to disk under
``.media_cache.json`` next to the dataset) so repeated runs and the
evaluation script don't re-pay OCR/ASR/API cost.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class MediaResult:
    media_type: str  # "image" | "voice"
    media_id: str
    file_path: str
    source: str  # "claude_vision" | "ocr" | "asr" | "metadata_only" | "missing"
    text: str = ""  # OCR transcript / ASR transcript
    caption: str = ""  # short natural-language description
    risk_tags: Optional[list] = None
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


SCAM_IMAGE_HINTS = re.compile(
    r"(otp|one[-\s]?time password|verify (your )?account|account (will be )?block|"
    r"login now|reset your password|urgent action required|click (the )?link|"
    r"claim (your )?(prize|reward|refund)|congratulations you (have )?won|"
    r"limited time|act now|payment failed|update kyc|suspend(ed)?)",
    re.IGNORECASE,
)

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


def _load_prompt(filename: str, fallback: str) -> str:
    """Load a prompt from prompts/<filename> (auditable/editable outside the
    code); fall back to the inline constant if the file can't be read."""
    try:
        with open(os.path.join(_PROMPTS_DIR, filename), "r", encoding="utf-8") as fh:
            text = fh.read().strip()
            return text or fallback
    except Exception:
        return fallback


class MediaCache:
    """Tiny JSON-file cache keyed by (media_type, media_id) so re-runs are cheap."""

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._data = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
            except Exception:
                self._data = {}

    def key(self, media_type: str, media_id: str) -> str:
        return f"{media_type}:{media_id}"

    def get(self, media_type: str, media_id: str) -> Optional[dict]:
        return self._data.get(self.key(media_type, media_id))

    def set(self, media_type: str, media_id: str, value: dict) -> None:
        self._data[self.key(media_type, media_id)] = value

    def save(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, sort_keys=True)
        except Exception:
            pass


class MediaAnalyzer:
    """Facade that tries Claude vision, then local OCR/ASR, then metadata-only."""

    def __init__(self, use_llm: bool = True, cache: Optional[MediaCache] = None, verbose: bool = False):
        self.use_llm = use_llm and bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.cache = cache
        self.verbose = verbose
        self._client = None
        if self.use_llm:
            try:
                import anthropic  # type: ignore

                self._client = anthropic.Anthropic()
            except Exception:
                self._client = None
                self.use_llm = False

    # -- public API -----------------------------------------------------
    def analyze(self, media_type: str, media_id: str, file_path: Optional[str]) -> MediaResult:
        if not media_id:
            return MediaResult(media_type=media_type, media_id="", file_path="", source="missing")

        cached = self.cache.get(media_type, media_id) if self.cache else None
        if cached:
            return MediaResult(**cached)

        if not file_path or not os.path.exists(file_path):
            result = MediaResult(
                media_type=media_type,
                media_id=media_id,
                file_path=file_path or "",
                source="missing",
                error="media file not found on disk",
            )
        elif media_type == "image":
            result = self._analyze_image(media_id, file_path)
        elif media_type == "voice":
            result = self._analyze_voice(media_id, file_path)
        else:
            result = MediaResult(media_type=media_type, media_id=media_id, file_path=file_path, source="metadata_only")

        if self.cache:
            self.cache.set(media_type, media_id, result.to_dict())
        return result

    # -- images -----------------------------------------------------------
    def _analyze_image(self, media_id: str, file_path: str) -> MediaResult:
        if self.use_llm and self._client is not None:
            try:
                return self._analyze_image_claude(media_id, file_path)
            except Exception as exc:  # network / quota / api errors -> degrade gracefully
                if self.verbose:
                    print(f"[media] Claude vision failed for {media_id}: {exc}")
        try:
            return self._analyze_image_ocr(media_id, file_path)
        except Exception as exc:
            if self.verbose:
                print(f"[media] local OCR failed for {media_id}: {exc}")
        return MediaResult(
            media_type="image",
            media_id=media_id,
            file_path=file_path,
            source="metadata_only",
            caption="image attachment (no OCR/vision backend available in this environment)",
        )

    def _analyze_image_claude(self, media_id: str, file_path: str) -> MediaResult:
        mime, _ = mimetypes.guess_type(file_path)
        mime = mime or "image/jpeg"
        with open(file_path, "rb") as fh:
            b64 = base64.standard_b64encode(fh.read()).decode("ascii")

        prompt = _load_prompt(
            "image_analysis.txt",
            (
                "You are analyzing one image attachment from a WhatsApp message for a "
                "notification-routing system. Reply with ONLY a compact JSON object with "
                "keys: \"ocr_text\" (any text visible in the image, transcribed as-is, "
                "empty string if none), \"caption\" (one sentence describing the image - "
                "e.g. promotional poster, screenshot of an app/chat, invoice/receipt, "
                "school circular, product photo, ID/QR code, safety/scam-style urgency "
                "graphic), and \"risk_tags\" (a list of short tags from this fixed set "
                "only if applicable: \"scam_style\", \"payment_request\", \"otp_or_verification\", "
                "\"promotional\", \"official_looking\", \"low_quality_or_spoofed\"). "
                "Do not include any text outside the JSON object."
            ),
        )
        resp = self._client.messages.create(
            model=os.environ.get("ROUTER_VISION_MODEL", "claude-sonnet-5"),
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        raw = "".join(getattr(block, "text", "") for block in resp.content)
        parsed = _extract_json(raw)
        ocr_text = str(parsed.get("ocr_text", "")) if parsed else ""
        caption = str(parsed.get("caption", "")) if parsed else raw.strip()[:300]
        risk_tags = parsed.get("risk_tags") if parsed else None
        if not isinstance(risk_tags, list):
            risk_tags = []
        return MediaResult(
            media_type="image",
            media_id=media_id,
            file_path=file_path,
            source="claude_vision",
            text=ocr_text,
            caption=caption,
            risk_tags=risk_tags,
        )

    def _analyze_image_ocr(self, media_id: str, file_path: str) -> MediaResult:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore

        img = Image.open(file_path)
        text = pytesseract.image_to_string(img) or ""
        text = text.strip()
        risk_tags = ["scam_style"] if SCAM_IMAGE_HINTS.search(text) else []
        caption = "image with OCR-extracted text" if text else "image attachment (no legible text found by OCR)"
        return MediaResult(
            media_type="image",
            media_id=media_id,
            file_path=file_path,
            source="ocr",
            text=text,
            caption=caption,
            risk_tags=risk_tags,
        )

    # -- voice notes --------------------------------------------------------
    def _analyze_voice(self, media_id: str, file_path: str) -> MediaResult:
        try:
            return self._analyze_voice_asr(media_id, file_path)
        except Exception as exc:
            if self.verbose:
                print(f"[media] local ASR failed for {media_id}: {exc}")
        size = 0
        try:
            size = os.path.getsize(file_path)
        except OSError:
            pass
        return MediaResult(
            media_type="voice",
            media_id=media_id,
            file_path=file_path,
            source="metadata_only",
            caption=f"voice note attachment ({size} bytes; no ASR backend available in this environment)",
        )

    def _analyze_voice_asr(self, media_id: str, file_path: str) -> MediaResult:
        import speech_recognition as sr  # type: ignore
        from pydub import AudioSegment  # type: ignore

        wav_path = file_path
        cleanup = False
        if not file_path.lower().endswith(".wav"):
            audio = AudioSegment.from_file(file_path)
            wav_path = file_path + ".tmp.wav"
            audio.export(wav_path, format="wav")
            cleanup = True
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        finally:
            if cleanup and os.path.exists(wav_path):
                os.remove(wav_path)

        risk_tags = ["scam_style"] if SCAM_IMAGE_HINTS.search(text) else []
        return MediaResult(
            media_type="voice",
            media_id=media_id,
            file_path=file_path,
            source="asr",
            text=text,
            caption="voice note transcribed via local speech recognition",
            risk_tags=risk_tags,
        )


def _extract_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None
