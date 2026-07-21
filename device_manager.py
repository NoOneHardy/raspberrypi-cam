from typing import List

from spotipy import Spotify

from spotipy_types.device import Device
from spotipy_types.playback_state import PlaybackState


class DeviceManager:
    def __init__(self, spotify: Spotify):
        self.spotify = spotify

    def get_first_available_device(self) -> Device | None:
        devices = self.spotify.devices()
        return devices[0] if len(devices) > 0 else None

    def get_active_device(self, playback: PlaybackState | None) -> Device | None:
        return playback['device'] if playback else None

    def get_playable_device(self, playback: PlaybackState | None) -> Device | None:
        active_device = self.get_active_device(playback)
        if active_device is not None:
            return active_device

        return self.get_first_available_device()

    def list_devices(self) -> List[Device]:
        return self.spotify.devices()
