"""Level 1: fetching only the needed label files from a remote zip, by byte range.

Run against a local HTTP server that honours ``Range``, serving a small archive laid out like
TotalSegmentator v1: per-subject CT volumes that must never be downloaded, label files, and a
``meta.csv``. Nothing leaves the machine.
"""

from __future__ import annotations

import io
import re
import threading
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("nibabel")
pytest.importorskip("numpy")

from assets.fetch import (  # noqa: E402
    BudgetExceededError,
    HttpRangeFile,
    RangeNotSupportedError,
    fetch_subset,
)
from assets.labels import load_label_map  # noqa: E402
from assets.phantom import write_phantom  # noqa: E402

CT_BYTES = 3 * 1024 * 1024


@pytest.fixture(scope="module")
def archive(tmp_path_factory) -> bytes:
    """Four subjects: two thorax scans, one with an empty heart, one abdomen-only scan."""
    root = tmp_path_factory.mktemp("dataset")
    write_phantom(root, name="s0001")
    write_phantom(root, name="s0002", empty=("heart_ventricle_left",))
    write_phantom(root, name="s0003")
    write_phantom(root, name="s0004")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(
            "Totalsegmentator_dataset/meta.csv",
            "image_id;age;gender;institute;study_type;split\n"
            "s0001;60;m;A;ct thorax;train\n"
            "s0002;55;f;A;ct thorax-abdomen;train\n"
            "s0003;70;m;B;ct abdomen;train\n"
            "s0004;48;f;B;ct whole body;val\n",
        )
        for subject in ("s0001", "s0002", "s0003", "s0004"):
            zf.writestr(f"Totalsegmentator_dataset/{subject}/ct.nii.gz", b"\x01" * CT_BYTES)
            for label_file in sorted((root / subject / "segmentations").iterdir()):
                zf.write(label_file, f"Totalsegmentator_dataset/{subject}/segmentations/"
                                     f"{label_file.name}")
            zf.writestr(f"Totalsegmentator_dataset/{subject}/segmentations/spleen.nii.gz",
                        b"\x02" * 50_000)
    return buffer.getvalue()


def _serve(payload: bytes, *, honour_range: bool) -> tuple[ThreadingHTTPServer, list[int]]:
    served: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002, D102
            pass

        def do_GET(self) -> None:  # noqa: N802, D102
            match = re.match(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
            if honour_range and match:
                start, end = int(match[1]), min(int(match[2]), len(payload) - 1)
                body = payload[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            else:
                body = payload
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
                served.append(len(body))
            except (BrokenPipeError, ConnectionResetError):
                served.append(-1)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, served


@pytest.fixture()
def remote(archive: bytes) -> Iterator[tuple[str, list[int]]]:
    """The archive behind a range-honouring server."""
    server, served = _serve(archive, honour_range=True)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/dataset.zip", served
    finally:
        server.shutdown()
        server.server_close()


class TestHttpRangeFile:
    """A seekable remote file."""

    def test_any_slice_reads_back_exactly(self, remote, archive: bytes) -> None:
        """Seek and read anywhere, across block boundaries, and get the archive's bytes."""
        url, _ = remote
        f = HttpRangeFile(url, budget_bytes=len(archive), block_size=4096)
        assert f.size == len(archive)
        for start, length in ((0, 10), (4090, 20), (len(archive) - 7, 50), (100_000, 70_000)):
            f.seek(start)
            assert f.read(length) == archive[start : start + length]

    def test_the_budget_is_enforced_before_fetching(self, remote, archive: bytes) -> None:
        """A read that would cross the budget raises and fetches nothing."""
        url, _ = remote
        f = HttpRangeFile(url, budget_bytes=10_000, block_size=4096)
        f.seek(0)
        with pytest.raises(BudgetExceededError):
            f.read(50_000)
        assert f.counter.used <= 10_000

    def test_a_server_without_ranges_is_refused(self, archive: bytes) -> None:
        """A 200 answer to a range request would be the whole archive: refuse it."""
        server, _ = _serve(archive, honour_range=False)
        try:
            with pytest.raises(RangeNotSupportedError):
                HttpRangeFile(f"http://127.0.0.1:{server.server_address[1]}/x.zip",
                              budget_bytes=len(archive))
        finally:
            server.shutdown()
            server.server_close()


class TestFetchSubset:
    """The whole selection."""

    def test_only_usable_thorax_subjects_are_fetched(self, remote, tmp_path: Path) -> None:
        """s0002 has an empty heart, s0003 is an abdomen scan; s0001 and s0004 qualify."""
        url, _ = remote
        report = fetch_subset(tmp_path, url=url, subjects=5, budget_bytes=20 * 1024 * 1024)
        assert report.subjects == ["s0001", "s0004"]
        assert "empty" in report.rejected["s0002"]
        assert "s0003" not in report.subjects

    def test_only_the_needed_labels_land_on_disk(self, remote, archive: bytes,
                                                 tmp_path: Path) -> None:
        """Exactly the label map's labels, byte-identical, and nothing else."""
        url, _ = remote
        fetch_subset(tmp_path, url=url, subjects=1, budget_bytes=20 * 1024 * 1024)
        names = sorted(p.name for p in (tmp_path / "s0001" / "segmentations").iterdir())
        assert names == sorted(f"{label}.nii.gz" for label in load_label_map().labels_needed())
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            original = zf.read("Totalsegmentator_dataset/s0001/segmentations/trachea.nii.gz")
        assert (tmp_path / "s0001" / "segmentations" / "trachea.nii.gz").read_bytes() == original

    def test_the_ct_volumes_are_never_downloaded(self, remote, archive: bytes,
                                                 tmp_path: Path) -> None:
        """Four 3 MB CTs sit in the archive; the whole fetch stays below one of them."""
        url, _ = remote
        report = fetch_subset(tmp_path, url=url, subjects=5, budget_bytes=20 * 1024 * 1024)
        assert report.bytes_downloaded < CT_BYTES
        assert report.archive_bytes == len(archive)

    def test_the_report_carries_source_and_license(self, remote, tmp_path: Path) -> None:
        """ADR-L1-004: provenance is written next to the data."""
        url, _ = remote
        fetch_subset(tmp_path, url=url, subjects=1, budget_bytes=20 * 1024 * 1024)
        text = (tmp_path / "fetch_report.json").read_text(encoding="utf-8")
        assert "CC BY 4.0" in text and url in text
