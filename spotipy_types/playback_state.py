from typing import TypedDict

from spotipy_types.device import Device


class PlaybackState(TypedDict):
    device: Device
    shuffle_state: bool
    is_playing: bool
    item: Track | Episode | None
    progress_ms: int


class Track(TypedDict):
    name: str

class Episode(TypedDict):
    name: str