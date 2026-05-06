"""
In-memory conversation history.
Stores last N turns per session. No database — resets on server restart.
Sufficient for demo purposes. Replace with persistent storage in v2.
"""

from collections import defaultdict

# session_id → list of message dicts
_history: dict[str, list[dict]] = defaultdict(list)

MAX_TURNS = 4   # keep last 4 turns (8 messages)


def get_history(session_id: str) -> list[dict]:
    """Return conversation history for a session."""
    return _history[session_id]


def add_turn(session_id: str, user_message: str, assistant_message: str):
    """Append a user/assistant turn to history."""
    history = _history[session_id]
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_message})

    # Trim to MAX_TURNS — keep most recent
    if len(history) > MAX_TURNS * 2:
        _history[session_id] = history[-(MAX_TURNS * 2):]


def clear_history(session_id: str):
    """Clear history for a session."""
    _history[session_id] = []