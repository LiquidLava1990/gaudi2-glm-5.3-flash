"""The decode key table against the readings c1c took (cells r2, 2026-11-08), which the r1 table of 19/23/31/47 MiB per row refused at its first key.

c1c flags-off peaked at 752.1-753.2 MiB on 16x8192, past 47 x 16 = 752.0, on all four ranks of both cells (c1c-off, c1c-det), and the boot was refused
on every rank: "raise DECODE_KEY_MIB_PER_ROW[8192] to 48 or more". Those two boots never reached the other 19 keys. For them there are two readings:
c1c with the levers on (/mnt/data1/claude-long/run1/k128, a full lattice, hbm_delta and peak per key) and c1b flags-off (hbm_delta only, six boots).
The transient the peak holds over the residual is 0.4 MiB at one row and 10.8-11.5 at sixteen; c1b's residual plus that transient is the projection
for c1c flags-off, and it came out 0.5 MiB above the one key both have (753.7 projected, 753.2 read). CPU only; the readings are the logs' own."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_decode_lattice as lattice  # noqa: E402
from test_decode_lattice import MIB  # noqa: E402

WIDTHS = (1024, 2048, 4096, 8192)
ROWS = (1, 2, 4, 8, 16)
# 16x8192 peak over the pre-capture in-use, MiB, per rank 0-3.
C1C_OFF_PEAKS = (752.1, 752.8, 752.7, 752.7)
C1C_DET_PEAKS = (752.8, 752.3, 753.1, 753.2)
OLD_8192_PER_ROW = 47
# Per width, one entry per row count in ROWS: max over ranks, MiB.
K128_LEVERS_ON_PEAK = {
    1024: (16.2, 36.3, 71.9, 143.0, 291.8), 2048: (16.4, 41.2, 80.3, 158.4, 321.2),
    4096: (23.1, 51.0, 97.1, 189.5, 380.8), 8192: (35.3, 62.9, 117.4, 242.9, 552.5),
}
C1B_FLAGS_OFF_RESIDUAL = {
    1024: (15.7, 36.3, 71.9, 143.2, 292.6), 2048: (15.7, 44.3, 86.6, 171.1, 346.6),
    4096: (25.5, 60.3, 115.8, 227.1, 455.3), 8192: (43.8, 92.3, 174.3, 338.4, 742.2),
}
K128_TRANSIENT = {
    1024: (0.4, 1.4, 2.7, 5.4, 10.8), 2048: (0.4, 1.4, 2.8, 5.4, 10.8),
    4096: (0.4, 1.5, 2.8, 5.5, 10.8), 8192: (0.5, 1.5, 3.0, 5.8, 11.5),
}
MIN_ROOM_MIB = 2.0  # the projection's tightest key (a two-row key) has 2.2; the unmeasured spread between boots of one key is under 1.2


class C1cFlagsOffPeaks(lattice.GraphCase):
    def peak_error(self, rows: int, width: int, peak_mib: float) -> str:
        return self.graph._peak_error((rows, width), int(peak_mib * MIB))

    def test_the_old_table_refused_every_logged_peak_and_named_48(self) -> None:
        with mock.patch.dict(self.graph.DECODE_KEY_MIB_PER_ROW, {8192: OLD_8192_PER_ROW}):
            self.assertEqual(self.graph.decode_key_estimate_bytes(16, 8192), 752 * MIB)
            for peak in C1C_OFF_PEAKS + C1C_DET_PEAKS:
                with self.subTest(peak=peak):
                    self.assertIn("raise DECODE_KEY_MIB_PER_ROW[8192] to 48 or more", self.peak_error(16, 8192, peak))

    def test_the_table_admits_every_logged_peak_at_16x8192_with_room_to_spare(self) -> None:
        estimate = self.graph.decode_key_estimate_bytes(16, 8192)
        for peak in C1C_OFF_PEAKS + C1C_DET_PEAKS:
            with self.subTest(peak=peak):
                self.assertEqual(self.peak_error(16, 8192, peak), "")
        self.assertGreaterEqual(estimate - int(max(C1C_OFF_PEAKS + C1C_DET_PEAKS) * MIB), 14 * MIB)

    def test_752_8_over_752_0_the_logged_case(self) -> None:
        """c1c-off rank 1 and c1c-det rank 0: 752.8 MiB over an estimate of 752.0."""
        with mock.patch.dict(self.graph.DECODE_KEY_MIB_PER_ROW, {8192: OLD_8192_PER_ROW}):
            self.assertIn("peaked at 752.8 MiB", self.peak_error(16, 8192, 752.8))
            self.assertIn("past its 752.0 MiB estimate", self.peak_error(16, 8192, 752.8))
        self.assertEqual(self.peak_error(16, 8192, 752.8), "")

    def test_every_key_of_the_levers_on_k128_boot_is_admitted(self) -> None:
        for width, peaks in K128_LEVERS_ON_PEAK.items():
            for rows, peak in zip(ROWS, peaks):
                with self.subTest(key=f"{rows}x{width}"):
                    self.assertEqual(self.peak_error(rows, width, peak), "")

    def test_c1b_residual_plus_the_k128_transient_keeps_room_on_every_key(self) -> None:
        """The projection for c1c flags-off, the keys its refused boots never reached."""
        for width in WIDTHS:
            for rows, residual, transient in zip(ROWS, C1B_FLAGS_OFF_RESIDUAL[width], K128_TRANSIENT[width]):
                with self.subTest(key=f"{rows}x{width}"):
                    room = self.graph.decode_key_estimate_bytes(rows, width) / MIB - (residual + transient)
                    self.assertGreaterEqual(room, MIN_ROOM_MIB)

    def test_the_extra_reserve_is_one_mib_per_row_and_allocates_nothing(self) -> None:
        """The lattice charge across the 20 keys rises by 124 MiB (31 rows x 4 widths) against the old table, and the ballast does not follow it."""
        old = {1024: 19, 2048: 23, 4096: 31, 8192: 47}
        with mock.patch.dict(self.graph.DECODE_KEY_MIB_PER_ROW, old):
            before = sum(self.graph.decode_key_estimate_bytes(*key) for key in self.graph.decode_keys())
        after = sum(self.graph.decode_key_estimate_bytes(*key) for key in self.graph.decode_keys())
        self.assertEqual((after - before) // MIB, 124)
        self.assertEqual(self.graph.SEAL_BALLAST_MIB, 512)


if __name__ == "__main__":
    unittest.main()
