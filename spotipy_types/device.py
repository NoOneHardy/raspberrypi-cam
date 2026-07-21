from typing import TypedDict


class Device(TypedDict):
    id: str
    is_active: bool
    name: str
    volume_percent: int
    supports_volume: bool