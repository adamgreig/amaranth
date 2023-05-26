"""
Utilities for computing cyclic redundancy checks (CRCs) in software and in
hardware.

Many commonly used CRC algorithms are available with their parameters
predefined in the ``Catalog`` class, while most other CRC algorithms can
be used by directly specifying their parameters in the ``Parameters`` class.
"""

from .. import *


__all__ = ["Parameters", "Processor", "Predefined", "Catalog"]


class Parameters:
    """
    Parameters for a CRC computation.

    The parameter set is based on the Williams model from
    "A Painless Guide to CRC Error Detection Algorithms":
    http://www.ross.net/crc/download/crc_v3.txt

    For a reference of standard CRC parameter sets, refer to:

    * `reveng`_'s catalogue, which uses an identical parameterisation,
    * `crcmod`_'s list of predefined functions, but remove the leading '1'
      from the polynominal, XOR the "Init-value" with "XOR-out" to obtain
      ``initial_crc``, and where "Reversed" is True, set both ``reflect_input``
      and ``reflect_output`` to True,
    * `CRC Zoo`_, which contains only polynomials; use the "explicit +1"
      form of polynomial but remove the leading '1'.

    .. _reveng: https://reveng.sourceforge.io/crc-catalogue/all.htm
    .. _crcmod: http://crcmod.sourceforge.net/crcmod.predefined.html
    .. _CRC Zoo: https://users.ece.cmu.edu/~koopman/crc/

    All entries from the reveng catalogue are available from the ``Catalog``
    class in this module.

    Parameters
    ----------
    crc_width : int
        Bit width of CRC word. Also known as "width" in the Williams model.
    data_width : int
        Bit width of data words.
    polynomial : int
        CRC polynomial to use, ``crc_width`` bits long, without the implicit
        ``x**crc_width`` term. Polynomial is always specified with the highest
        order terms in the most significant bit positions; use
        ``reflect_input`` and ``reflect_output`` to perform a least
        significant bit first computation.
    initial_crc : int
        Initial value of CRC register at reset. Most significant bit always
        corresponds to the highest order term in the CRC register.
    reflect_input : bool
        If True, the input data words are bit-reflected, so that they are
        processed least significant bit first.
    reflect_output : bool
        If True, the output CRC is bit-reflected, so the least-significant bit
        of the output is the highest-order bit of the CRC register.
        Note that this reflection is performed over the entire CRC register;
        for transmission you may want to treat the output as a little-endian
        multi-word value, so for example the reflected 16-bit output 0x4E4C
        would be transmitted as the two octets 0x4C 0x4E, each transmitted
        least significant bit first.
    xor_output : int
        The output CRC will be the CRC register XOR'd with this value, applied
        after any output bit-reflection.
    """
    def __init__(self, crc_width, data_width, polynomial, initial_crc,
                 reflect_input, reflect_output, xor_output):
        self.crc_width = int(crc_width)
        self.data_width = int(data_width)
        self.polynomial = int(polynomial)
        self.initial_crc = int(initial_crc)
        self.reflect_input = bool(reflect_input)
        self.reflect_output = bool(reflect_output)
        self.xor_output = int(xor_output)

        assert self.crc_width > 0
        assert self.data_width > 0
        assert self.polynomial < 2 ** self.crc_width
        assert self.initial_crc < 2 ** self.crc_width
        assert self.xor_output < 2 ** self.crc_width

    def residue(self):
        """
        Compute the residue value for this CRC, which is the value left in the
        CRC register after processing any valid codeword.
        """
        # Residue is computed by initialising to (possibly reflected)
        # xor_output, feeding crc_width worth of 0 bits, then taking
        # the (possibly reflected) output without any XOR.
        if self.reflect_output:
            init = self._reflect(self.xor_output, self.crc_width)
        else:
            init = self.xor_output
        crc = Parameters(
            self.crc_width, self.crc_width, self.polynomial, init, False,
            self.reflect_output, 0)
        return crc.compute([0])

    def check(self):
        """
        Compute the CRC of the ASCII string "123456789", commonly used as
        a verification value for CRC parameters.
        """
        return self.compute(b"123456789")

    def create(self):
        """
        Returns a ``Processor`` configured with these parameters.
        """
        return Processor(self)

    def compute(self, data):
        """
        Computes and returns the CRC of all data words in ``data``.
        """
        # Precompute some constants we use every iteration.
        word_max = (1 << self.data_width) - 1
        top_bit = 1 << (self.crc_width + self.data_width - 1)
        crc_mask = (1 << (self.crc_width + self.data_width)) - 1
        poly_shifted = self.polynomial << self.data_width

        # Implementation notes:
        # We always compute most-significant bit first, which means the
        # polynomial and initial value may be used as-is, and the reflect_in
        # and reflect_out values have their usual sense.
        # However, when computing word-at-a-time and MSbit-first, we must align
        # the input word so its MSbit lines up with the MSbit of the previous
        # CRC value. When the CRC width is smaller than the word width, this
        # would normally truncate data bits.
        # Instead, we shift the initial CRC left by the data width, and the
        # data word left by the crc width, lining up their MSbits no matter
        # the relation between the two widths.
        # The new CRC is then shifted right by the data width before output.

        crc = self.initial_crc << self.data_width
        for word in data:
            assert 0 <= word <= word_max

            if self.reflect_input:
                word = self._reflect(word, self.data_width)

            crc ^= word << self.crc_width
            for _ in range(self.data_width):
                if crc & top_bit:
                    crc = (crc << 1) ^ poly_shifted
                else:
                    crc <<= 1
            crc &= crc_mask

        crc >>= self.data_width
        if self.reflect_output:
            crc = self._reflect(crc, self.crc_width)

        crc ^= self.xor_output
        return crc

    @staticmethod
    def _reflect(word, n):
        """
        Bitwise-reflects an n-bit word ``word``.
        """
        return int(f"{word:0{n}b}"[::-1], 2)

    def _matrices(self):
        """
        Computes the F and G matrices for parallel CRC computation, treating
        the CRC as a linear time-invariant system described by the state
        relation x(t+1) = F.x(i) + G.u(i), where x(i) and u(i) are column
        vectors of the bits of the CRC register and input word, F is the n-by-n
        matrix relating the old state to the new state, and G is the n-by-m
        matrix relating the new data to the new state, where n is the CRC
        width and m is the data word width.

        The matrices are ordered least-significant-bit first; in other words
        the first entry, with index (0, 0), corresponds to the effect of the
        least-significant bit of the input on the least-significant bit of the
        output.

        For convenience of implementation, both matrices are returned
        transposed: the first index is the input bit, and the second index is
        the corresponding output bit.

        The matrices are used to select which bits are XORd together to compute
        each bit i of the new state: if F[j][i] is set then bit j of the old
        state is included in the XOR, and if G[j][i] is set then bit j of the
        new data is included.

        These matrices are not affected by ``initial_crc``, ``reflect_input``,
        ``reflect__output``, or ``xor_output``.
        """
        f = []
        g = []
        crc = Parameters(
            self.crc_width, self.data_width, self.polynomial, 0, False, False, 0)
        for i in range(self.crc_width):
            crc.initial_crc = 2 ** i
            w = crc.compute([0])
            f.append([int(x) for x in reversed(f"{w:0{self.crc_width}b}")])
        for i in range(self.data_width):
            crc.initial_crc = 0
            w = crc.compute([2 ** i])
            g.append([int(x) for x in reversed(f"{w:0{self.crc_width}b}")])
        return f, g

    def __repr__(self):
        return f"Parameters(crc_width={self.crc_width}," \
               f" data_width={self.data_width}," \
               f" polynomial=0x{self.polynomial:0{self.crc_width//4}x}," \
               f" initial_crc=0x{self.initial_crc:0{self.crc_width//4}x}," \
               f" reflect_input={self.reflect_input}," \
               f" reflect_output={self.reflect_output}," \
               f" xor_output=0x{self.xor_output:0{self.crc_width//4}x})"


class Predefined:
    """
    Predefined CRC parameters.

    Each instance of this class is a standard or well-known CRC algorithm,
    specified except for the data word width. Refer to the ``Parameters``
    class for the meaning of the parameters, and additionally:

    * ``check`` is the CRC of the ASCII byte string "123456789", commonly
      used to validate a CRC algorithm.
    * ``residue`` is the expected CRC value after processing a valid
      codeword (a data word followed immediately by its CRC).

    To create a ``Parameters`` instance, call the ``Predefined`` object
    with the required data word width, which defaults to 8 bits.
    """
    def __init__(self, crc_width, polynomial, initial_crc, reflect_input,
                 reflect_output, xor_output, check, residue):
        self.crc_width = crc_width
        self.polynomial = polynomial
        self.initial_crc = initial_crc
        self.reflect_input = reflect_input
        self.reflect_output = reflect_output
        self.xor_output = xor_output
        self.check = check
        self.residue = residue

    def __call__(self, data_width=8):
        return Parameters(self.crc_width, data_width, self.polynomial,
                          self.initial_crc, self.reflect_input, self.reflect_output,
                          self.xor_output)

    def __repr__(self):
        return f"Predefined(crc_width={self.crc_width}," \
               f" polynomial=0x{self.polynomial:0{self.crc_width//4}x}," \
               f" initial_crc=0x{self.initial_crc:0{self.crc_width//4}x}," \
               f" reflect_input={self.reflect_input}," \
               f" reflect_output={self.reflect_output}," \
               f" xor_output=0x{self.xor_output:0{self.crc_width//4}x}," \
               f" check=0x{self.check:0{self.crc_width//4}x}," \
               f" residue=0x{self.residue:0{self.crc_width//4}x})"


class Catalog:
    """
    Catalog of predefined CRC algorithms.

    All entries are from `reveng`_, accessed on 2023-05-25.

    To use an entry, call it with an optional ``data_width`` which defaults
    to 8. For example::

        crc8 = m.submodules.crc8 = crc.Catalog.CRC8_AUTOSAR().create()

    """
    CRC3_GSM = Predefined(3, 0x3, 0x0, False, False, 0x7, 0x4, 0x2)
    CRC3_ROHC = Predefined(3, 0x3, 0x7, True, True, 0x0, 0x6, 0x0)
    CRC4_G_704 = CRC4_ITU = Predefined(4, 0x3, 0x0, True, True, 0x0, 0x7, 0x0)
    CRC4_INTERLAKEN = Predefined(4, 0x3, 0xf, False, False, 0xf, 0xb, 0x2)
    CRC5_EPC_C1G2 = CRC5_EPC = Predefined(5, 0x09, 0x09, False, False, 0x00, 0x00, 0x00)
    CRC5_G_704 = CRC5_ITU = Predefined(5, 0x15, 0x00, True, True, 0x00, 0x07, 0x00)
    CRC5_USB = Predefined(5, 0x05, 0x1f, True, True, 0x1f, 0x19, 0x06)
    CRC6_CDMA2000_A = Predefined(6, 0x27, 0x3f, False, False, 0x00, 0x0d, 0x00)
    CRC6_CDMA2000_B = Predefined(6, 0x07, 0x3f, False, False, 0x00, 0x3b, 0x00)
    CRC6_DARC = Predefined(6, 0x19, 0x00, True, True, 0x00, 0x26, 0x00)
    CRC6_G_704 = CRC6_ITU = Predefined(6, 0x03, 0x00, True, True, 0x00, 0x06, 0x00)
    CRC6_GSM = Predefined(6, 0x2f, 0x00, False, False, 0x3f, 0x13, 0x3a)
    CRC7_MMC = Predefined(7, 0x09, 0x00, False, False, 0x00, 0x75, 0x00)
    CRC7_ROHC = Predefined(7, 0x4f, 0x7f, True, True, 0x00, 0x53, 0x00)
    CRC7_UMTS = Predefined(7, 0x45, 0x00, False, False, 0x00, 0x61, 0x00)
    CRC8_AUTOSAR = Predefined(8, 0x2f, 0xff, False, False, 0xff, 0xdf, 0x42)
    CRC8_BLUETOOTH = Predefined(8, 0xa7, 0x00, True, True, 0x00, 0x26, 0x00)
    CRC8_CDMA2000 = Predefined(8, 0x9b, 0xff, False, False, 0x00, 0xda, 0x00)
    CRC8_DARC = Predefined(8, 0x39, 0x00, True, True, 0x00, 0x15, 0x00)
    CRC8_DVB_S2 = Predefined(8, 0xd5, 0x00, False, False, 0x00, 0xbc, 0x00)
    CRC8_GSM_A = Predefined(8, 0x1d, 0x00, False, False, 0x00, 0x37, 0x00)
    CRC8_GSM_B = Predefined(8, 0x49, 0x00, False, False, 0xff, 0x94, 0x53)
    CRC8_HITAG = Predefined(8, 0x1d, 0xff, False, False, 0x00, 0xb4, 0x00)
    CRC8_I_432_1 = CRC8_ITU = Predefined(8, 0x07, 0x00, False, False, 0x55, 0xa1, 0xac)
    CRC8_I_CODE = Predefined(8, 0x1d, 0xfd, False, False, 0x00, 0x7e, 0x00)
    CRC8_LTE = Predefined(8, 0x9b, 0x00, False, False, 0x00, 0xea, 0x00)
    CRC8_MAXIM_DOW = CRC8_MAXIM = Predefined(8, 0x31, 0x00, True, True, 0x00, 0xa1, 0x00)
    CRC8_MIFARE_MAD = Predefined(8, 0x1d, 0xc7, False, False, 0x00, 0x99, 0x00)
    CRC8_NRSC_5 = Predefined(8, 0x31, 0xff, False, False, 0x00, 0xf7, 0x00)
    CRC8_OPENSAFETY = Predefined(8, 0x2f, 0x00, False, False, 0x00, 0x3e, 0x00)
    CRC8_ROHC = Predefined(8, 0x07, 0xff, True, True, 0x00, 0xd0, 0x00)
    CRC8_SAE_J1850 = Predefined(8, 0x1d, 0xff, False, False, 0xff, 0x4b, 0xc4)
    CRC8_SMBUS = Predefined(8, 0x07, 0x00, False, False, 0x00, 0xf4, 0x00)
    CRC8_TECH_3250 = CRC8_AES = CRC8_ETU = Predefined(8, 0x1d, 0xff, True, True, 0x00, 0x97, 0x00)
    CRC8_WCDMA = Predefined(8, 0x9b, 0x00, True, True, 0x00, 0x25, 0x00)
    CRC10_ATM = CRC8_I_610 = Predefined(10, 0x233, 0x000, False, False, 0x000, 0x199, 0x000)
    CRC10_CDMA2000 = Predefined(10, 0x3d9, 0x3ff, False, False, 0x000, 0x233, 0x000)
    CRC10_GSM = Predefined(10, 0x175, 0x000, False, False, 0x3ff, 0x12a, 0x0c6)
    CRC11_FLEXRAY = Predefined(11, 0x385, 0x01a, False, False, 0x000, 0x5a3, 0x000)
    CRC11_UMTS = Predefined(11, 0x307, 0x000, False, False, 0x000, 0x061, 0x000)
    CRC12_CDMA2000 = Predefined(12, 0xf13, 0xfff, False, False, 0x000, 0xd4d, 0x000)
    CRC12_DECT = Predefined(12, 0x80f, 0x000, False, False, 0x000, 0xf5b, 0x000)
    CRC12_GSM = Predefined(12, 0xd31, 0x000, False, False, 0xfff, 0xb34, 0x178)
    CRC12_UMTS = CRC12_3GPP = Predefined(12, 0x80f, 0x000, False, True, 0x000, 0xdaf, 0x000)
    CRC13_BBC = Predefined(13, 0x1cf5, 0x0000, False, False, 0x0000, 0x04fa, 0x0000)
    CRC14_DARC = Predefined(14, 0x0805, 0x0000, True, True, 0x0000, 0x082d, 0x0000)
    CRC14_GSM = Predefined(14, 0x202d, 0x0000, False, False, 0x3fff, 0x30ae, 0x031e)
    CRC15_CAN = Predefined(15, 0x4599, 0x0000, False, False, 0x0000, 0x059e, 0x0000)
    CRC15_MPT1327 = Predefined(15, 0x6815, 0x0000, False, False, 0x0001, 0x2566, 0x6815)
    CRC16_ARC = CRC16_IBM = Predefined(16, 0x8005, 0x0000, True, True, 0x0000, 0xbb3d, 0x0000)
    CRC16_CDMA2000 = Predefined(16, 0xc867, 0xffff, False, False, 0x0000, 0x4c06, 0x0000)
    CRC16_CMS = Predefined(16, 0x8005, 0xffff, False, False, 0x0000, 0xaee7, 0x0000)
    CRC16_DDS_110 = Predefined(16, 0x8005, 0x800d, False, False, 0x0000, 0x9ecf, 0x0000)
    CRC16_DECT_R = Predefined(16, 0x0589, 0x0000, False, False, 0x0001, 0x007e, 0x0589)
    CRC16_DECT_X = Predefined(16, 0x0589, 0x0000, False, False, 0x0000, 0x007f, 0x0000)
    CRC16_DNP = Predefined(16, 0x3d65, 0x0000, True, True, 0xffff, 0xea82, 0x66c5)
    CRC16_EN_13757 = Predefined(16, 0x3d65, 0x0000, False, False, 0xffff, 0xc2b7, 0xa366)
    CRC16_GENIBUS = CRC16_DARC = CRC16_EPC = CRC16_EPC_C1G2 = CRC16_I_CODE = \
        Predefined(16, 0x1021, 0xffff, False, False, 0xffff, 0xd64e, 0x1d0f)
    CRC16_GSM = Predefined(16, 0x1021, 0x0000, False, False, 0xffff, 0xce3c, 0x1d0f)
    CRC16_IBM_3740 = CRC16_AUTOSAR = CRC16_CCITT_FALSE = \
        Predefined(16, 0x1021, 0xffff, False, False, 0x0000, 0x29b1, 0x0000)
    CRC16_IBM_SDLC = CRC16_ISO_HDLC = CRC16_ISO_IEC_14443_3_B = CRC16_X25 = \
        Predefined(16, 0x1021, 0xffff, True, True, 0xffff, 0x906e, 0xf0b8)
    CRC16_ISO_IEC_14443_3_A = Predefined(16, 0x1021, 0xc6c6, True, True, 0x0000, 0xbf05, 0x0000)
    CRC16_KERMIT = CRC16_BLUETOOTH = CRC16_CCITT = CRC16_CCITT_TRUE = CRC16_V_41_LSB = \
        Predefined(16, 0x1021, 0x0000, True, True, 0x0000, 0x2189, 0x0000)
    CRC16_LJ1200 = Predefined(16, 0x6f63, 0x0000, False, False, 0x0000, 0xbdf4, 0x0000)
    CRC16_M17 = Predefined(16, 0x5935, 0xffff, False, False, 0x0000, 0x772b, 0x0000)
    CRC16_MAXIM_DOW = CRC16_MAXIM = Predefined(16, 0x8005, 0x0000, True, True, 0xffff, 0x44c2, 0xb001)
    CRC16_MCRF4XX = Predefined(16, 0x1021, 0xffff, True, True, 0x0000, 0x6f91, 0x0000)
    CRC16_MODBUS = Predefined(16, 0x8005, 0xffff, True, True, 0x0000, 0x4b37, 0x0000)
    CRC16_NRSC_5 = Predefined(16, 0x080b, 0xffff, True, True, 0x0000, 0xa066, 0x0000)
    CRC16_OPENSAFETY_A = Predefined(16, 0x5935, 0x0000, False, False, 0x0000, 0x5d38, 0x0000)
    CRC16_OPENSAFETY_B = Predefined(16, 0x755b, 0x0000, False, False, 0x0000, 0x20fe, 0x0000)
    CRC16_PROFIBUS = CRC16_IEC_61158_2 = Predefined(16, 0x1dcf, 0xffff, False, False, 0xffff, 0xa819, 0xe394)
    CRC16_RIELLO = Predefined(16, 0x1021, 0xb2aa, True, True, 0x0000, 0x63d0, 0x0000)
    CRC16_SPI_FUJITSU = CRC16_AUG_CCITT = Predefined(16, 0x1021, 0x1d0f, False, False, 0x0000, 0xe5cc, 0x0000)
    CRC16_T10_DIF = Predefined(16, 0x8bb7, 0x0000, False, False, 0x0000, 0xd0db, 0x0000)
    CRC16_TELEDISK = Predefined(16, 0xa097, 0x0000, False, False, 0x0000, 0x0fb3, 0x0000)
    CRC16_TMS37157 = Predefined(16, 0x1021, 0x89ec, True, True, 0x0000, 0x26b1, 0x0000)
    CRC16_UMTS = CRC16_BUYPASS = CRC16_VERIFONE = \
        Predefined(16, 0x8005, 0x0000, False, False, 0x0000, 0xfee8, 0x0000)
    CRC16_USB = Predefined(16, 0x8005, 0xffff, True, True, 0xffff, 0xb4c8, 0xb001)
    CRC16_XMODEM = CRC16_ACORN = CRC16_LTE = CRC16_V_41_MSB = CRC16_ZMODEM = \
        Predefined(16, 0x1021, 0x0000, False, False, 0x0000, 0x31c3, 0x0000)
    CRC17_CAN_FD = Predefined(17, 0x1685b, 0x00000, False, False, 0x00000, 0x04f03, 0x00000)
    CRC21_CAN_FD = Predefined(21, 0x102899, 0x000000, False, False, 0x000000, 0x0ed841, 0x000000)
    CRC24_BLE = Predefined(24, 0x00065b, 0x555555, True, True, 0x000000, 0xc25a56, 0x000000)
    CRC24_FLEXRAY_A = Predefined(24, 0x5d6dcb, 0xfedcba, False, False, 0x000000, 0x7979bd, 0x000000)
    CRC24_FLEXRAY_B = Predefined(24, 0x5d6dcb, 0xabcdef, False, False, 0x000000, 0x1f23b8, 0x000000)
    CRC24_INTERLAKEN = Predefined(24, 0x328b63, 0xffffff, False, False, 0xffffff, 0xb4f3e6, 0x144e63)
    CRC24_LTE_A = Predefined(24, 0x864cfb, 0x000000, False, False, 0x000000, 0xcde703, 0x000000)
    CRC24_LTE_B = Predefined(24, 0x800063, 0x000000, False, False, 0x000000, 0x23ef52, 0x000000)
    CRC24_OPENPGP = Predefined(24, 0x864cfb, 0xb704ce, False, False, 0x000000, 0x21cf02, 0x000000)
    CRC24_OS_9 = Predefined(24, 0x800063, 0xffffff, False, False, 0xffffff, 0x200fa5, 0x800fe3)
    CRC30_CDMA = Predefined(30, 0x2030b9c7, 0x3fffffff, False, False, 0x3fffffff, 0x04c34abf, 0x34efa55a)
    CRC31_PHILIPS = Predefined(31, 0x04c11db7, 0x7fffffff, False, False, 0x7fffffff, 0x0ce9e46c, 0x4eaf26f1)
    CRC32_AIXM = Predefined(32, 0x814141ab, 0x00000000, False, False, 0x00000000, 0x3010bf7f, 0x00000000)
    CRC32_AUTOSAR = Predefined(32, 0xf4acfb13, 0xffffffff, True, True, 0xffffffff, 0x1697d06a, 0x904cddbf)
    CRC32_BASE91_D = Predefined(32, 0xa833982b, 0xffffffff, True, True, 0xffffffff, 0x87315576, 0x45270551)
    CRC32_BZIP2 = CRC32_AAL5 = CRC32_DECT_B = \
        Predefined(32, 0x04c11db7, 0xffffffff, False, False, 0xffffffff, 0xfc891918, 0xc704dd7b)
    CRC32_CD_ROM_EDC = Predefined(32, 0x8001801b, 0x00000000, True, True, 0x00000000, 0x6ec2edc4, 0x00000000)
    CRC32_CKSUM = CRC32_POSIX = \
        Predefined(32, 0x04c11db7, 0x00000000, False, False, 0xffffffff, 0x765e7680, 0xc704dd7b)
    CRC32_ISCSI = CRC32_BASE91_C = CRC32_CASTAGNOLI = CRC32_INTERLAKEN = \
        Predefined(32, 0x1edc6f41, 0xffffffff, True, True, 0xffffffff, 0xe3069283, 0xb798b438)
    CRC32_ISO_HDLC = CRC32_ADCCP = CRC32_V_42 = CRC32_XZ = CRC32_PKZIP = CRC32_ETHERNET = \
        Predefined(32, 0x04c11db7, 0xffffffff, True, True, 0xffffffff, 0xcbf43926, 0xdebb20e3)
    CRC32_JAMCRC = Predefined(32, 0x04c11db7, 0xffffffff, True, True, 0x00000000, 0x340bc6d9, 0x00000000)
    CRC32_MEF = Predefined(32, 0x741b8cd7, 0xffffffff, True, True, 0x00000000, 0xd2c22f51, 0x00000000)
    CRC32_MPEG_2 = Predefined(32, 0x04c11db7, 0xffffffff, False, False, 0x00000000, 0x0376e6e7, 0x00000000)
    CRC32_XFER = Predefined(32, 0x000000af, 0x00000000, False, False, 0x00000000, 0xbd0be338, 0x00000000)
    CRC40_GSM = Predefined(40, 0x0004820009, 0x0000000000, False, False, 0xffffffffff, 0xd4164fc646, 0xc4ff8071ff)
    CRC64_ECMA_182 = Predefined(
        64, 0x42f0e1eba9ea3693, 0x0000000000000000, False, False,
        0x0000000000000000, 0x6c40df5f0b497347, 0x0000000000000000)
    CRC64_GO_ISO = Predefined(
        64, 0x000000000000001b, 0xffffffffffffffff, True, True,
        0xffffffffffffffff, 0xb90956c775a41001, 0x5300000000000000)
    CRC64_MS = Predefined(
        64, 0x259c84cba6426349, 0xffffffffffffffff, True, True,
        0x0000000000000000, 0x75d4b74f024eceea, 0x0000000000000000)
    CRC64_REDIS = Predefined(
        64, 0xad93d23594c935a9, 0x0000000000000000, True, True,
        0x0000000000000000, 0xe9c6d914c4b8d9ca, 0x0000000000000000)
    CRC64_WE = Predefined(
        64, 0x42f0e1eba9ea3693, 0xffffffffffffffff, False, False,
        0xffffffffffffffff, 0x62ec59e3f1a4f00a, 0xfcacbebd5931a992)
    CRC64_XZ = CRC64_ECMA = Predefined(
        64, 0x42f0e1eba9ea3693, 0xffffffffffffffff, True, True,
        0xffffffffffffffff, 0x995dc9bbdf1939fa, 0x49958c9abd7d353f)
    CRC82_DARC = Predefined(
        82, 0x0308c0111011401440411, 0x000000000000000000000, True, True,
        0x000000000000000000000, 0x09ea83f625023801fd612, 0x000000000000000000000)


class Processor(Elaboratable):
    """
    Cyclic redundancy check (CRC) processor module.

    This module generates CRCs from an input data stream, which can be used
    to validate an existing CRC or generate a new CRC. It is configured by
    the ``Parameters`` class, which can handle most forms of CRCs. Refer to
    that class's documentation for a description of the parameters.

    The CRC value is updated on any clock cycle where ``valid`` is asserted,
    with the updated value available on the ``crc`` output on the subsequent
    clock cycle. The latency is therefore one clock cycle, and the throughput
    is one data word per clock cycle.

    The CRC is reset to its initial value whenever ``first`` is asserted.
    ``first`` and ``valid`` may be asserted on the same clock cycle, in which
    case a new CRC computation is started with the current value of ``data``.

    With ``data_width=1``, a classic bit-serial CRC is implemented for the
    given polynomial in a Galois-type shift register. For larger values of
    ``data_width``, a similar architecture computes every new bit of the
    CRC in parallel.

    The ``match_detected`` output may be used to validate data with a trailing
    CRC (also known as a codeword). If the most recently processed word(s) form
    the valid CRC of all the previous data since ``first`` was asserted, the
    CRC register will always take on a fixed value known as the residue.  The
    ``match_detected`` output indicates whether the CRC register currently
    contains this residue.

    Parameters
    ----------
    parameters : Parameters
        CRC parameters.

    Attributes
    ----------
    first : Signal(), in
        Assert to indicate the start of a CRC computation, re-initialising
        the CRC register to the initial value. May be asserted simultaneously
        with ``valid``.
    data : Signal(data_width), in
        Data word to add to CRC when ``valid`` is asserted.
    valid : Signal(), in
        Assert when ``data`` is valid to add the data word to the CRC.
    crc : Signal(crc_width), out
        Registered CRC output value, updated one clock cycle after ``valid``
        becomes asserted.
    match_detected : Signal(), out
        Asserted if the current CRC value indicates a valid codeword has been
        received.
    """
    def __init__(self, parameters):
        assert isinstance(parameters, Parameters)
        self.crc_width = parameters.crc_width
        self.data_width = parameters.data_width
        self.polynomial = parameters.polynomial
        self.initial_crc = Const(parameters.initial_crc, self.crc_width)
        self.reflect_input = parameters.reflect_input
        self.reflect_output = parameters.reflect_output
        self.xor_output = parameters.xor_output
        self._matrix_f, self._matrix_g = parameters._matrices()
        self._residue = parameters.residue()

        self.first = Signal()
        self.data = Signal(self.data_width)
        self.valid = Signal()
        self.crc = Signal(self.crc_width)
        self.match_detected = Signal()

    def elaborate(self, platform):
        m = Module()

        crc_reg = Signal(self.crc_width, reset=self.initial_crc.value)
        data_in = Signal(self.data_width)

        # Optionally bit-reflect input words.
        if self.reflect_input:
            m.d.comb += data_in.eq(self.data[::-1])
        else:
            m.d.comb += data_in.eq(self.data)

        # Optionally bit-reflect and then XOR the output.
        if self.reflect_output:
            m.d.comb += self.crc.eq(crc_reg[::-1] ^ self.xor_output)
        else:
            m.d.comb += self.crc.eq(crc_reg ^ self.xor_output)

        # Compute next CRC state.
        source = Mux(self.first, self.initial_crc, crc_reg)
        with m.If(self.valid):
            for i in range(self.crc_width):
                bit = 0
                for j in range(self.crc_width):
                    if self._matrix_f[j][i]:
                        bit ^= source[j]
                for j in range(self.data_width):
                    if self._matrix_g[j][i]:
                        bit ^= data_in[j]
                m.d.sync += crc_reg[i].eq(bit)
        with m.Elif(self.first):
            m.d.sync += crc_reg.eq(self.initial_crc)

        # Check for residue match, indicating a valid codeword.
        if self.reflect_output:
            m.d.comb += self.match_detected.eq(crc_reg[::-1] == self._residue)
        else:
            m.d.comb += self.match_detected.eq(crc_reg == self._residue)

        return m
