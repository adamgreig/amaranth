import unittest

from amaranth.sim import *
from amaranth.lib.crc import Parameters, Catalog


# Subset of catalogue CRCs used to test the hardware CRC implementation,
# as testing all algorithms takes a long time for little benefit.
# Selected to ensure coverage of CRC width, initial value, reflection, and
# XOR output.
CRCS = [
    Catalog.CRC3_GSM, Catalog.CRC4_INTERLAKEN, Catalog.CRC5_USB,
    Catalog.CRC8_AUTOSAR, Catalog.CRC8_BLUETOOTH, Catalog.CRC8_I_432_1,
    Catalog.CRC12_UMTS, Catalog.CRC15_CAN, Catalog.CRC16_ARC,
    Catalog.CRC16_IBM_3740, Catalog.CRC17_CAN_FD, Catalog.CRC21_CAN_FD,
    Catalog.CRC24_FLEXRAY_A, Catalog.CRC32_AUTOSAR, Catalog.CRC32_BZIP2,
    Catalog.CRC32_ISO_HDLC, Catalog.CRC40_GSM
]


class CRCTestCase(unittest.TestCase):
    def test_checks(self):
        """
        Verify computed check values and residues match catalogue entries.
        """
        for name in dir(Catalog):
            if name.startswith("CRC"):
                crc = getattr(Catalog, name)
                assert crc().check() == crc.check
                assert crc().residue() == crc.residue

    def test_crc_bytes(self):
        """
        Verify CRC generation by computing the check value for each CRC
        in the catalogue with byte-sized inputs.
        """
        for predefined in CRCS:
            p = predefined(data_width=8)
            crc = p.create()

            def process():
                for word in b"123456789":
                    yield crc.first.eq(word == b"1")
                    yield crc.data.eq(word)
                    yield crc.valid.eq(1)
                    yield
                yield crc.valid.eq(0)
                yield
                self.assertEqual((yield crc.crc), predefined.check)

            sim = Simulator(crc)
            sim.add_sync_process(process)
            sim.add_clock(1e-6)
            sim.run()

    def test_crc_words(self):
        """
        Verify CRC generation for non-byte-sized data by computing a check
        value for 1, 2, 4, 16, 32, and 64-bit inputs.
        """
        # We can't use the catalogue check value since it requires 8-bit
        # inputs, so we'll instead use an input of b"12345678".
        data = b"12345678"
        # Split data into individual bits. When input is reflected, we have
        # to reflect each byte first, then form the input words, then let
        # the CRC module reflect those words, to get the same effective input.
        bits = "".join(f"{x:08b}" for x in data)
        bits_r = "".join(f"{x:08b}"[::-1] for x in data)

        for predefined in CRCS:
            for m in (1, 2, 4, 16, 32, 64):
                p = predefined(data_width=m)
                crc = p.create()
                # Use a SoftwareCRC with byte inputs to compute new checks.
                swcrc = predefined(data_width=8)
                check = swcrc.compute(data)
                # Chunk input bits into m-bit words, reflecting if needed.
                if predefined.reflect_input:
                    d = [bits_r[i : i+m][::-1] for i in range(0, len(bits), m)]
                else:
                    d = [bits[i : i+m] for i in range(0, len(bits), m)]
                words = [int(x, 2) for x in d]

                def process():
                    yield crc.first.eq(1)
                    yield
                    yield crc.first.eq(0)
                    for word in words:
                        yield crc.data.eq(word)
                        yield crc.valid.eq(1)
                        yield
                    yield crc.valid.eq(0)
                    yield
                    self.assertEqual((yield crc.crc), check)

                sim = Simulator(crc)
                sim.add_sync_process(process)
                sim.add_clock(1e-6)
                sim.run()

    def test_crc_match(self):
        """Verify match_detected output detects valid codewords."""
        for predefined in CRCS:
            n = predefined.crc_width
            m = 8 if n % 8 == 0 else 1
            p = predefined(data_width=m)
            check = predefined.check
            crc = p.create()

            if m == 8:
                # For CRCs which are multiples of one byte wide, we can easily
                # append the correct checksum in bytes.
                check_b = check.to_bytes(n // 8, "little" if p.reflect_output else "big")
                words = b"123456789" + check_b
            else:
                # For other CRC sizes, use single-bit input data.
                if p.reflect_output:
                    check_b = check.to_bytes((n + 7)//8, "little")
                    if not p.reflect_input:
                        # For cross-endian CRCs, flip the CRC bits separately.
                        check_b = bytearray(int(f"{x:08b}"[::-1], 2) for x in check_b)
                else:
                    shift = 8 - (n % 8)
                    check_b = (check << shift).to_bytes((n + 7)//8, "big")
                    # No catalogue CRCs have ref_in but not ref_out.
                codeword = b"123456789" + check_b
                words = []
                for byte in codeword:
                    if predefined.reflect_input:
                        words += [int(x) for x in f"{byte:08b}"[::-1]]
                    else:
                        words += [int(x) for x in f"{byte:08b}"]
                words = words[:72 + n]

            def process():
                yield crc.first.eq(1)
                yield
                yield crc.first.eq(0)
                for word in words:
                    yield crc.data.eq(word)
                    yield crc.valid.eq(1)
                    yield
                yield crc.valid.eq(0)
                yield
                self.assertTrue((yield crc.match_detected))

            sim = Simulator(crc)
            sim.add_sync_process(process)
            sim.add_clock(1e-6)
            sim.run()
