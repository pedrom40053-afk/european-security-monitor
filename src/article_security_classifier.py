"""Educational keyword topic tagger.

This introductory example is not a semantic classifier, validation system,
operational security decision, or production model.
"""
from __future__ import annotations

TOPIC_TERMS = {
    "Cybersecurity": ("cyberattack", "ransomware", "data breach", "cybersecurity"),
    "Energy": ("energy", "electricity", "gas supply", "power grid"),
    "Defence": ("defence", "defense", "military", "armed forces"),
    "Diplomacy": ("diplomacy", "summit", "foreign minister", "peace talks"),
}

def classify_topic(title: str) -> str:
    """Return a coarse illustrative label for a headline."""
    text = str(title or "").casefold()
    for topic, words in TOPIC_TERMS.items():
        if any(word in text for word in words):
            return topic
    return "Other / unclassified"
