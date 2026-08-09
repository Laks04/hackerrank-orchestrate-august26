"""Message Notification Router.

A personalized, multimodal WhatsApp notification router.

Pipeline (see pipeline.py for orchestration):
  1. data_loader   - load every dataset/*.csv file into indexed, joinable tables
  2. media         - understand image/voice attachments (VLM/OCR/ASR, graceful degrade)
  3. features      - fuse message + user + group + business + history signals
  4. rules         - deterministic scoring engine (works with zero API access)
  5. llm           - optional Claude reasoning layer that refines the rules
                     verdict, guarded so it can never override a hard safety call
  6. evidence      - retrieve historical message_ids that justify the decision
"""

__all__ = [
    "data_loader",
    "media",
    "features",
    "rules",
    "llm",
    "evidence",
    "pipeline",
    "schema",
]
