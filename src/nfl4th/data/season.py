from __future__ import annotations

from datetime import date


def current_target_season(today: date) -> int:
    # NFL season S runs from kickoff in September of year S through the
    # Super Bowl in February of year S+1, so any date before September
    # belongs to the season that started the previous calendar year.
    return today.year if today.month >= 9 else today.year - 1
