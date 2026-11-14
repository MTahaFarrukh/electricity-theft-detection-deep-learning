"""Extract the SGCC dataset from its split zip archive (data.z01, data.z02, data.zip).

This is a *spanned* zip: the central directory (in data.zip) stores each
member's disk number and an offset relative to the start of that disk.
Python's zipfile ignores disk numbers, so it cannot open the parts even when
they are concatenated. Instead, this script reads the central directory with
zipfile, then reads each member's bytes from the parts as one continuous
stream and decompresses them itself. No 7-Zip installation is needed.

    python scripts/extract_sgcc.py                       # uses data/raw/sgcc_zip -> data/raw
    python scripts/extract_sgcc.py --parts-dir <dir> --out-dir <dir>
"""

import argparse
import struct
import sys
import zipfile
import zlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PART_ORDER = ["data.z01", "data.z02", "data.zip"]  # disk 0, 1, 2
LOCAL_HEADER = struct.Struct("<4s2B4HL2L2H")
LOCAL_HEADER_SIGNATURE = b"PK\x03\x04"
CHUNK = 1 << 20


class SpannedReader:
    """Read the archive parts as if they were one file."""

    def __init__(self, parts: list[Path]):
        self.parts = parts
        self.sizes = [p.stat().st_size for p in parts]

    def global_offset(self, disk: int, offset: int) -> int:
        return sum(self.sizes[:disk]) + offset

    def read(self, start: int, length: int):
        """Yield `length` bytes starting at global offset `start`, chunk by chunk."""
        for part, size in zip(self.parts, self.sizes):
            if start >= size:
                start -= size
                continue
            with part.open("rb") as f:
                f.seek(start)
                while length > 0 and (block := f.read(min(CHUNK, length))):
                    length -= len(block)
                    yield block
            start = 0
            if length == 0:
                return
        if length:
            raise EOFError("Archive parts end before the member's data does")


def extract_member(reader: SpannedReader, member: zipfile.ZipInfo, target: Path) -> None:
    header_start = reader.global_offset(member.volume, member.header_offset)
    header = b"".join(reader.read(header_start, LOCAL_HEADER.size))
    fields = LOCAL_HEADER.unpack(header)
    if fields[0] != LOCAL_HEADER_SIGNATURE:
        raise ValueError(f"No local file header for {member.filename} at offset {header_start}")
    name_len, extra_len = fields[-2], fields[-1]
    data_start = header_start + LOCAL_HEADER.size + name_len + extra_len

    if member.compress_type == zipfile.ZIP_DEFLATED:
        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    elif member.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"Unsupported compression type {member.compress_type} for {member.filename}")
    else:
        decompressor = None

    crc = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        for block in reader.read(data_start, member.compress_size):
            data = decompressor.decompress(block) if decompressor else block
            crc = zlib.crc32(data, crc)
            out.write(data)
        if decompressor:
            tail = decompressor.flush()
            crc = zlib.crc32(tail, crc)
            out.write(tail)

    if crc != member.CRC or target.stat().st_size != member.file_size:
        target.unlink()
        raise ValueError(f"CRC/size check failed for {member.filename}; the archive parts may be corrupt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "sgcc_zip")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    args = parser.parse_args()

    parts = [args.parts_dir / name for name in PART_ORDER]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        print(f"ERROR: missing archive parts in {args.parts_dir}: {missing}")
        return 1

    reader = SpannedReader(parts)
    out_root = args.out_dir.resolve()
    with zipfile.ZipFile(parts[-1]) as archive:  # the last part holds the central directory
        members = archive.infolist()

    for member in members:
        if member.is_dir():
            continue
        target = (out_root / member.filename).resolve()
        # Refuse paths that would escape the output folder.
        if out_root not in target.parents:
            print(f"Skipping unsafe path in archive: {member.filename}")
            continue
        extract_member(reader, member, target)
        print(f"Extracted {member.filename} ({member.file_size / 1e6:.1f} MB, CRC verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
