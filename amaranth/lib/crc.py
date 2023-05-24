"""
Utilities for computing cyclic redundancy checks (CRCs) in software and in
hardware.
"""

from .. import *


__all__ = ["Parameters", "Processor", "SoftwareProcessor"]


class Parameters:
    """
    Parameters for a CRC computation.

    The parameter set is based on the Williams model from
    "A Painless Guide to CRC Error Detection Algorithms":
    http://www.ross.net/crc/download/crc_v3.txt

    For a reference of standard CRC parameter sets, refer to:

    * `reveng`_'s catalogue, which uses an identical parameterisation,
    * `crcmod`_'s list of predefined functions, but remove the leading '1'
      from the polynominal and where "Reversed" is True, set both
      ``reflect_input`` and ``reflect_output`` to True,
    * `CRC Zoo`_, which contains only polynomials; use the "explicit +1"
      form of polynomial but remove the leading '1'.

    .. _reveng: https://reveng.sourceforge.io/crc-catalogue/all.htm
    .. _crcmod: http://crcmod.sourceforge.net/crcmod.predefined.html
    .. _CRC Zoo: https://users.ece.cmu.edu/~koopman/crc/

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
            init = SoftwareProcessor._reflect(self.xor_output, self.crc_width)
        else:
            init = self.xor_output
        crc = self.create_software()
        crc.initial_crc = init
        crc.data_width = self.crc_width
        crc.reflect_input = False
        crc.xor_output = 0
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

    def create_software(self):
        """
        Returns a ``SoftwareProcessor`` configured with these parameters.
        """
        return SoftwareProcessor(self)

    def compute(self, data):
        """
        Computes and returns the CRC of all data words in ``data``.
        """
        return SoftwareProcessor(self).compute(data)

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
        crc = self.create_software()
        crc.reflect_input = crc.reflect_output = False
        crc.xor_output = 0
        for i in range(self.crc_width):
            crc.initial_crc = 2 ** i
            w = crc.compute([0])
            f.append([int(x) for x in reversed(f"{w:0{self.crc_width}b}")])
        for i in range(self.data_width):
            crc.initial_crc = 0
            w = crc.compute([2 ** i])
            g.append([int(x) for x in reversed(f"{w:0{self.crc_width}b}")])
        return f, g


class SoftwareProcessor:
    """
    Compute CRCs in software.
    """
    def __init__(self, parameters):
        assert isinstance(parameters, Parameters)
        self.crc_width = parameters.crc_width
        self.data_width = parameters.data_width
        self.polynomial = parameters.polynomial
        self.initial_crc = parameters.initial_crc
        self.reflect_input = parameters.reflect_input
        self.reflect_output = parameters.reflect_output
        self.xor_output = parameters.xor_output

    def compute(self, data):
        """
        Compute in software the CRC of the input data words in ``data``,
        using all CRC parameters.

        Returns the final CRC value.
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
        with m.If(self.valid):
            for i in range(self.crc_width):
                bit = 0
                for j in range(self.crc_width):
                    if self._matrix_f[j][i]:
                        bit ^= Mux(self.first, self.initial_crc[j], crc_reg[j])
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
