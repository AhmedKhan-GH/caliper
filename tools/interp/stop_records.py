"""Bounded stop records for offline dictionary commands."""


def stop_reason(error):
    """Return valid UTF-8 without NUL and with at most 1,024 bytes."""
    text = str(error).replace("\0", "")[:1024]
    return text.encode("utf-8", errors="replace")[:1024].decode("utf-8", errors="ignore")


def clean_stop_record(error):
    """Use one terminal record for startup and resource stops."""
    return {"stage": "clean_stop", "status": "interrupted", "reason": stop_reason(error)}
