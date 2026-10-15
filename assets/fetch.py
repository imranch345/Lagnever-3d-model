"""Fetch only the label files Level 1 needs from a remote zip, by HTTP byte ranges.

The TotalSegmentator v1.0 dataset is one 28.4 GB zip. Level 1 needs 14 small label volumes
for about 20 subjects. A zip keeps its index at the end, and Zenodo serves byte ranges, so
the standard library's ``zipfile`` can read the index and decompress single members through a
seekable file object that fetches ranges on demand. Nothing else is downloaded.

Guards, because a mistake here could pull tens of gigabytes:

* a hard byte budget -- exceeding it raises before the request is made;
* a server that ignores ``Range`` (answers 200 instead of 206) is refused, never read;
* the index is read once; members are taken only from subjects that ``meta.csv`` describes as
  covering the thorax, until enough subjects whose heart and lung labels are non-empty.

Standard library only. Data lands under ``datasets/raw/`` (git-ignored) with a report that
records what was fetched, from where, under which license (ADR-L1-004).

    python -m assets.fetch --out datasets/raw/totalsegmentator_v1 --subjects 20 --max-mb 250
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import struct
import time
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from assets.labels import DEFAULT_LABEL_MAP, LabelMap, load_label_map

__all__ = [
    "DATASET_URL",
    "BudgetExceededError",
    "HttpRangeFile",
    "RangeNotSupportedError",
    "fetch_subset",
]

DATASET_URL = "https://zenodo.org/records/6802614/files/Totalsegmentator_dataset.zip?download=1"
_MEMBER = re.compile(r"(?:^|/)(?P<subject>s\d{4})/segmentations/(?P<label>[a-z0-9_]+)\.nii\.gz$")

#: Labels whose emptiness disqualifies a subject: Level 1 needs the heart and both lungs.
REQUIRED_NON_EMPTY = (
    "heart_ventricle_left",
    "heart_atrium_right",
    "lung_upper_lobe_left",
    "lung_lower_lobe_right",
)


class RangeNotSupportedError(RuntimeError):
    """The server answered a range request with the whole file; refused, not read."""


class BudgetExceededError(RuntimeError):
    """A request would take the total download past the budget."""


@dataclass
class _Counter:
    budget: int
    used: int = 0
    requests: int = 0


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file, fetched in cached blocks by byte range."""

    def __init__(
        self,
        url: str,
        *,
        budget_bytes: int,
        block_size: int = 128 * 1024,
        max_cached_blocks: int = 64,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        super().__init__()
        self.url = url
        self.block_size = block_size
        self.max_cached_blocks = max_cached_blocks
        self.counter = _Counter(budget=budget_bytes)
        self._open = opener
        self._cache: dict[int, bytes] = {}
        self._position = 0
        self.size = self._probe_size()

    def _request(self, start: int, end: int) -> bytes:
        """Bytes ``start..end`` inclusive, or raise; never accepts a full-file answer."""
        length = end - start + 1
        if self.counter.used + length > self.counter.budget:
            raise BudgetExceededError(
                f"fetching {length} more bytes would exceed the {self.counter.budget}-byte budget "
                f"({self.counter.used} used)"
            )
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with self._open(request, timeout=60) as response:
            if getattr(response, "status", None) != 206:
                raise RangeNotSupportedError(
                    f"{self.url} answered {getattr(response, 'status', '?')} to a range request"
                )
            data = bytes(response.read(length + 1))
        if len(data) != length:
            raise OSError(f"expected {length} bytes for {start}-{end}, got {len(data)}")
        self.counter.used += length
        self.counter.requests += 1
        return data

    def _probe_size(self) -> int:
        request = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with self._open(request, timeout=60) as response:
            if getattr(response, "status", None) != 206:
                raise RangeNotSupportedError(f"{self.url} does not serve byte ranges")
            content_range = response.headers.get("Content-Range", "")
            response.read(2)
        match = re.match(r"bytes \d+-\d+/(\d+)", content_range)
        if not match:
            raise RangeNotSupportedError(f"no total size in Content-Range {content_range!r}")
        self.counter.used += 1
        self.counter.requests += 1
        return int(match.group(1))

    # --- io.RawIOBase ---------------------------------------------------
    def readable(self) -> bool:
        """Always readable."""
        return True

    def seekable(self) -> bool:
        """Seeking is what makes zip index reads cheap."""
        return True

    def tell(self) -> int:
        """Current position."""
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move without fetching anything."""
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self.size}[whence]
        self._position = max(0, base + offset)
        return self._position

    def _block(self, index: int) -> bytes:
        if index not in self._cache:
            start = index * self.block_size
            end = min(start + self.block_size, self.size) - 1
            if len(self._cache) >= self.max_cached_blocks:
                self._cache.pop(next(iter(self._cache)))
            self._cache[index] = self._request(start, end)
        return self._cache[index]

    def readinto(self, buffer: Any) -> int:
        """Fill ``buffer`` from the current position.

        A long read (the zip index, a whole member) is one range request; short reads go
        through the block cache, so scattered small reads do not each cost a round trip.
        """
        view = memoryview(buffer).cast("B")
        wanted = min(len(view), max(0, self.size - self._position))
        if wanted > 4 * self.block_size:
            data = self._request(self._position, self._position + wanted - 1)
            view[:wanted] = data
            self._position += wanted
            return wanted
        done = 0
        while done < wanted:
            index, offset = divmod(self._position, self.block_size)
            block = self._block(index)
            take = min(wanted - done, len(block) - offset)
            view[done : done + take] = block[offset : offset + take]
            done += take
            self._position += take
        return done


@dataclass
class FetchReport:
    """What was fetched, recorded next to the data."""

    url: str
    license: str
    attribution: str
    subjects: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)
    members: int = 0
    bytes_downloaded: int = 0
    requests: int = 0
    archive_bytes: int = 0
    seconds: float = 0.0


def _thorax_subjects(archive: zipfile.ZipFile) -> list[str] | None:
    """Subjects whose ``meta.csv`` study type mentions the thorax, or ``None`` without one."""
    names = [n for n in archive.namelist() if n.endswith("meta.csv")]
    if not names:
        return None
    # utf-8-sig: the real meta.csv starts with a byte-order mark, which would otherwise
    # become part of the first column's name and hide every image_id.
    with archive.open(names[0]) as handle:
        text = io.TextIOWrapper(handle, encoding="utf-8-sig")
        rows = list(csv.DictReader(text, delimiter=";"))
    if rows and len(rows[0]) == 1:  # comma-separated after all
        with archive.open(names[0]) as handle:
            rows = list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig")))
    out = []
    for row in rows:
        study = " ".join(str(v) for k, v in row.items() if k and "study" in k.lower()).lower()
        subject = str(row.get("image_id", "")).strip()
        if subject and ("thorax" in study or "chest" in study or "whole body" in study):
            out.append(subject)
    return out


def _non_empty(raw: bytes) -> bool:
    """Whether a gzipped NIfTI-1 label holds any voxel, reading ``vox_offset`` from its header."""
    data = gzip.decompress(raw)
    order = "<" if struct.unpack("<i", data[:4])[0] == 348 else ">"
    offset = int(struct.unpack(f"{order}f", data[108:112])[0])
    return any(data[max(offset, 348):])


def fetch_subset(
    out_dir: Path,
    *,
    url: str = DATASET_URL,
    subjects: int = 20,
    budget_bytes: int = 250 * 1024 * 1024,
    label_map: LabelMap | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> FetchReport:
    """Fetch the needed labels for up to ``subjects`` usable subjects into ``out_dir``."""
    label_map = label_map or load_label_map(DEFAULT_LABEL_MAP)
    needed = set(label_map.labels_needed())
    started = time.time()
    remote = HttpRangeFile(url, budget_bytes=budget_bytes, opener=opener)
    report = FetchReport(
        url=url, license=label_map.source.license, attribution=label_map.source.attribution,
        archive_bytes=remote.size,
    )
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 16)) as archive:
        by_subject: dict[str, dict[str, zipfile.ZipInfo]] = {}
        for info in archive.infolist():
            match = _MEMBER.search(info.filename)
            if match and match["label"] in needed:
                by_subject.setdefault(match["subject"], {})[match["label"]] = info
        order = _thorax_subjects(archive)
        candidates = [s for s in (order if order is not None else sorted(by_subject))
                      if s in by_subject]
        for subject in candidates:
            if len(report.subjects) >= subjects:
                break
            members = by_subject[subject]
            if set(members) != needed:
                report.rejected[subject] = f"missing {sorted(needed - set(members))}"
                continue
            payload = {label: archive.read(members[label]) for label in sorted(needed)}
            empty = [label for label in REQUIRED_NON_EMPTY if not _non_empty(payload[label])]
            if empty:
                report.rejected[subject] = f"empty {empty}"
                continue
            folder = out_dir / subject / "segmentations"
            folder.mkdir(parents=True, exist_ok=True)
            for label, raw in payload.items():
                (folder / f"{label}.nii.gz").write_bytes(raw)
            report.subjects.append(subject)
            report.members += len(payload)
    report.bytes_downloaded = remote.counter.used
    report.requests = remote.counter.requests
    report.seconds = round(time.time() - started, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fetch_report.json").write_text(
        json.dumps(report.__dict__, indent=2), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch the Level 1 subset."""
    parser = argparse.ArgumentParser(description="Fetch Level 1 label files by byte range.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--max-mb", type=int, default=250)
    parser.add_argument("--url", default=DATASET_URL)
    args = parser.parse_args(argv)
    report = fetch_subset(
        args.out, url=args.url, subjects=args.subjects, budget_bytes=args.max_mb * 1024 * 1024
    )
    print(
        f"{len(report.subjects)} subjects, {report.members} files, "
        f"{report.bytes_downloaded / 1e6:.1f} MB in {report.requests} requests "
        f"of a {report.archive_bytes / 1e9:.1f} GB archive"
    )
    return 0 if report.subjects else 1


if __name__ == "__main__":
    raise SystemExit(main())
