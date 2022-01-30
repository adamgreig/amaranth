import unittest

from amaranth.sim import *
from amaranth.lib.crc import CRC, _SoftwareCRC


# Catalogue of standard CRCs used to test the CRC implementation.
# Selected for coverage of bit width, initial value, reflection, and XOR out.
# Contains (n, poly, init, ref_in, ref_out, xor_out, check, residue).
CRCS = [
    # CRC-3/GSM
    (3, 0x3, 0x0, False, False, 0x7, 0x4, 0x2),
    # CRC-4/INTERLAKEN
    (4, 0x3, 0xF, False, False, 0xF, 0xB, 0x2),
    # CRC-5/USB
    (5, 0x05, 0x1F, True, True, 0x1F, 0x19, 0x06),
    # CRC-8/AUTOSAR
    (8, 0x2F, 0xFF, False, False, 0xFF, 0xDF, 0x42),
    # CRC-8/BLUETOOTH
    (8, 0xA7, 0x00, True, True, 0x00, 0x26, 0x00),
    # CRC-8/I-432-1
    (8, 0x07, 0x00, False, False, 0x55, 0xA1, 0xAC),
    # CRC-12/UMTS
    (12, 0x80F, 0x000, False, True, 0x000, 0xDAF, 0x000),
    # CRC-15/CAN
    (15, 0x4599, 0x0000, False, False, 0x0000, 0x059E, 0x0000),
    # CRC-16/ARC
    (16, 0x8005, 0x0000, True, True, 0x0000, 0xBB3D, 0x0000),
    # CRC-16/IBM-3740 (aka CRC-16/AUTOSAR, CRC-16/CCITT-FALSE)
    (16, 0x1021, 0xFFFF, False, False, 0x0000, 0x29B1, 0x0000),
    # CRC-17/CAN-FD
    (17, 0x1685B, 0x00000, False, False, 0x00000, 0x04F03, 0x00000),
    # CRC-21/CAN-FD
    (21, 0x102899, 0x000000, False, False, 0x000000, 0x0ED841, 0x000000),
    # CRC-24/FLEXRAY-A
    (24, 0x5D6DCB, 0xFEDCBA, False, False, 0x000000, 0x7979BD, 0x000000),
    # CRC-32/AUTOSAR
    (32, 0xF4ACFB13, 0xFFFFFFFF, True, True, 0xFFFFFFFF, 0x1697D06A, 0x904CDDBF),
    # CRC-32/BZIP2
    (32, 0x04C11DB7, 0xFFFFFFFF, False, False, 0xFFFFFFFF, 0xFC891918, 0xC704DD7B),
    # CRC-32/ISO-HDLC (aka Ethernet)
    (32, 0x04C11DB7, 0xFFFFFFFF, True, True, 0xFFFFFFFF, 0xCBF43926, 0xDEBB20E3),
    # CRC-40/GSM
    (40, 0x0004820009, 0x0000000000, False, False, 0xFFFFFFFFFF, 0xD4164FC646, 0xC4FF8071FF),
]


class CRCTestCase(unittest.TestCase):
    def test_crc_bytes(self):
        """
        Verify CRC generation by computing the check value for each CRC
        in the catalogue with byte-sized inputs.
        """
        for n, poly, init, ref_in, ref_out, xor_out, check, _ in CRCS:
            crc = CRC(n, 8, poly, init, ref_in, ref_out, xor_out)

            def process():
                yield crc.rst.eq(1)
                yield
                yield crc.rst.eq(0)
                for word in b"123456789":
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
        # the CRC module reflect those worsd, to get the same effective input.
        bits = "".join(f"{x:08b}" for x in data)
        bits_r = "".join(f"{x:08b}"[::-1] for x in data)

        for n, poly, init, ref_in, ref_out, xor_out, _, _ in CRCS:
            for m in (1, 2, 4, 16, 32, 64):
                crc = CRC(n, m, poly, init, ref_in, ref_out, xor_out)
                # Use a SoftwareCRC with byte inputs to compute new checks.
                swcrc = _SoftwareCRC(n, 8, poly, init, ref_in, ref_out, xor_out)
                check = swcrc._compute(data)
                # Chunk input bits into m-bit words, reflecting if needed.
                if ref_in:
                    d = [bits_r[i : i+m][::-1] for i in range(0, len(bits), m)]
                else:
                    d = [bits[i : i+m] for i in range(0, len(bits), m)]
                words = [int(x, 2) for x in d]

                def process():
                    yield crc.rst.eq(1)
                    yield
                    yield crc.rst.eq(0)
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
        """Verify match output detects valid codewords."""
        for n, poly, init, ref_in, ref_out, xor_out, check, _ in CRCS:
            m = 8 if n % 8 == 0 else 1
            crc = CRC(n, m, poly, init, ref_in, ref_out, xor_out)

            if m == 8:
                # For CRCs which are multiples of one byte wide, we can easily
                # append the correct checksum in bytes.
                check_b = check.to_bytes(n // 8, "little" if ref_out else "big")
                words = b"123456789" + check_b
            else:
                # For other CRC sizes, use single-bit input data.
                if ref_out:
                    check_b = check.to_bytes((n + 7)//8, "little")
                    if not ref_in:
                        # For cross-endian CRCs, flip the CRC bits separately.
                        check_b = bytearray(int(f"{x:08b}"[::-1], 2) for x in check_b)
                else:
                    shift = 8 - (n % 8)
                    check_b = (check << shift).to_bytes((n + 7)//8, "big")
                    # No catalogue CRCs have ref_in but not ref_out.
                codeword = b"123456789" + check_b
                words = []
                for byte in codeword:
                    if ref_in:
                        words += [int(x) for x in f"{byte:08b}"[::-1]]
                    else:
                        words += [int(x) for x in f"{byte:08b}"]
                words = words[:72 + n]

            def process():
                yield crc.rst.eq(1)
                yield
                yield crc.rst.eq(0)
                for word in words:
                    yield crc.data.eq(word)
                    yield crc.valid.eq(1)
                    yield
                yield crc.valid.eq(0)
                yield
                self.assertTrue((yield crc.match))

            sim = Simulator(crc)
            sim.add_sync_process(process)
            sim.add_clock(1e-6)
            sim.run()
