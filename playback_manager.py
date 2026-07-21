from spotipy import Spotify

from device_manager import DeviceManager
from spotipy_types.playback_state import PlaybackState, Track, Episode


class PlaybackManager:
    device_manager: DeviceManager
    spotify: Spotify

    def __init__(self, spotify: Spotify):
        self.device_manager = DeviceManager(spotify)
        self.spotify = spotify

    def pause(self):
        self.spotify.pause_playback()

    def play(self):
        device = self.device_manager.get_playable_device(self.get_playback_state())
        if device is None:
            return

        self.spotify.start_playback(device['id'])
        item, playback = self.get_item()
        if item is None or playback is None:
            return
        if device['supports_volume']:
            print(f'Playing {item["name"]} on {device["name"]} at {device["volume_percent"]}% volume')
        else:
            print(f'Playing {item["name"]} on {device["name"]}')

    def get_playback_state(self) -> PlaybackState | None:
        return self.spotify.current_playback()

    def get_item(self) -> tuple[Track | Episode | None, PlaybackState | None]:
        playback = self.get_playback_state()
        return playback['item'] if playback else None, playback

    def is_playing(self) -> bool:
        playback = self.get_playback_state()
        return playback['is_playing'] if playback is not None else False

    def toggle_playback_state(self):
        playback = self.get_playback_state()
        if playback is None:
            print('No playback')
            return

        if self.is_playing():
            self.pause()
        else:
            self.play()
        pass

    def skip_track(self):
        if not self.is_playing():
            self.play()

        self.spotify.next_track()

    def replay(self):
        init_playback = self.get_playback_state()

        if not init_playback or not init_playback['is_playing']:
            self.play()

        item, playback = self.get_item()
        if item is None or playback is None:
            return

        if playback['progress_ms'] > 6000:
            self.spotify.seek_track(0)
            print(f'Restart track {item["name"]}')
        else:
            self.spotify.previous_track()
