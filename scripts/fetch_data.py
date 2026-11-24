"""Download and parse the one public dataset the real-data experiments use.

    uv run python scripts/fetch_data.py

The file is the daily Fama/French three-factor file from Kenneth French's data
library. It is free and needs no account or key. Only the market excess return
(``Mkt-RF``) is used. The zip is saved under ``data/``, which git ignores, and
is not committed.

The library rebuilds this file every month from a new CRSP release, and past
values are sometimes revised. ``reproduce.py`` uses rows up to
``SAMPLE_END`` only and writes a fingerprint of the series it used to
``results/data_fingerprint.csv``, so a reader can tell whether their download
matches the one behind the committed tables.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ZIP_PATH = DATA_DIR / "F-F_Research_Data_Factors_daily_CSV.zip"
SAMPLE_END = 20251231  # last date used; later rows are ignored


@dataclass(frozen=True)
class MarketSeries:
    dates: NDArray[np.int64]
    """Trading dates as yyyymmdd integers."""
    excess: NDArray[np.float64]
    """Daily market return minus the one-month T-bill return, as a decimal."""


def download(path: Path = ZIP_PATH) -> Path:
    """Fetch the zip unless it is already on disk."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "research-client/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = resp.read()
    zipfile.ZipFile(io.BytesIO(payload)).testzip()
    path.write_bytes(payload)
    return path


def parse_daily_factors(text: str, end: int = SAMPLE_END) -> MarketSeries:
    """Parse the CSV inside the zip. Values in the file are percentages.

    The file has free-text lines before the header and a copyright line after
    the table; only rows that start with an eight-digit date are kept.
    """
    dates: list[int] = []
    excess: list[float] = []
    header: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split(",")]
        if header is None:
            if len(cells) > 1 and cells[0] == "" and "Mkt-RF" in cells:
                header = cells
            continue
        if not (len(cells[0]) == 8 and cells[0].isdigit()):
            if dates:
                break  # end of the daily table
            continue
        date = int(cells[0])
        if date > end:
            break
        dates.append(date)
        excess.append(float(cells[header.index("Mkt-RF")]) / 100.0)
    if header is None or not dates:
        raise ValueError("no daily Mkt-RF table found")
    d = np.asarray(dates, dtype=np.int64)
    if np.any(np.diff(d) <= 0):
        raise ValueError("dates are not strictly increasing")
    return MarketSeries(dates=d, excess=np.asarray(excess, dtype=np.float64))


def load_market(path: Path = ZIP_PATH, end: int = SAMPLE_END) -> MarketSeries:
    """Download if needed, then parse."""
    download(path)
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(name).decode("latin-1")
    return parse_daily_factors(text, end)


def main() -> None:
    series = load_market()
    print(
        f"{ZIP_PATH.name}: {series.dates.size} trading days, "
        f"{series.dates[0]} to {series.dates[-1]}"
    )


if __name__ == "__main__":
    main()
