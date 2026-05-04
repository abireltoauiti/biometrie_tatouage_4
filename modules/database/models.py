"""modules/database/models.py — Data classes for Person and Event."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Person:
    id:     Optional[int]
    name:   str
    status: str   # 'authorized' | 'unauthorized'


@dataclass
class Event:
    id:           Optional[int]
    person_name:  str
    category:     str     # AUTHORIZED | UNAUTHORIZED | UNKNOWN
    camera_id:    str
    timestamp:    str
    image_path:   str
    hash_sha256:  str
    alert_level:  str     # LOG_ONLY | ALERT | ALARM
