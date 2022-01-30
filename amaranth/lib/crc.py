from .. import *


__all__ = ["CRC"]


class _SoftwareCRC:
    """
    Compute CRCs in software, used to create constants required by the ``CRC``
    class. Refer to its documentation for the meaning of each parameter.
    """
    def __init__(self, n, m, poly, init, ref_in, ref_out, xor_out):
        self.n = int(n)
        self.m = int(m)
        self.poly = int(poly)
        self.init = int(init)
        self.ref_in = bool(ref_in)
        self.ref_out = bool(ref_out)
        self.xor_out = int(xor_out)

        assert self.n > 0
        assert self.m > 0
        assert self.poly < 2 ** n
        assert self.init < 2 ** n
        assert self.xor_out < 2 ** n


    def _update(self, crc, word):
        """
        Run one round of CRC processing in software, given the current CRC
        value ``crc`` and the new data word ``word``.

        This method is not affected by ``init``, ``ref_in``, ``ref_out``,
        or ``xor_out`; those parameters are applied elsewhere.

        Returns the new value for ``crc``.
        """
        # Implementation notes:
        # We always compute most-significant bit first, which means the
        # polynomial and initial value may be used as-is, and the ref_in
        # and ref_out values have their usual sense.
        # However, when computing word-at-a-time and MSbit-first, we must
        # align the input word so its MSbit lines up with the MSbit of the
        # previous CRC value. When the CRC length n is smaller than the word
        # length m, this would normally truncate data bits.
        # Instead, we shift the previous CRC left by m and the word left by
        # n, lining up their MSbits no matter the relation between n and m.
        # The new CRC is then shifted right by m before output.
        assert 0 <= word < (2 ** self.m)
        crc = (crc << self.m) ^ (word << self.n)
        for _ in range(self.m):
            if (crc >> self.m) & (1 << (self.n - 1)):
                crc = (crc << 1) ^ (self.poly << self.m)
            else:
                crc <<= 1
        return (crc >> self.m) & ((2 ** self.n) - 1)

    def _compute(self, data):
        """
        Compute in software the CRC of the input data words in ``data``.

        Returns the final CRC value.
        """
        crc = self.init
        for word in data:
            if self.ref_in:
                word = self._reflect(word, self.m)
            crc = self._update(crc, word)
        if self.ref_out:
            crc = self._reflect(crc, self.n)
        crc ^= self.xor_out
        return crc

    def _residue(self):
        """
        Compute in software the residue for the configured CRC, which is the
        value left in the CRC register after processing a valid codeword.
        """
        # Residue is computed by initialising to (reflected) xor_out, feeding
        # n 0 bits, then taking the (reflected) output, without any XOR.
        if self.ref_out:
            init = self._reflect(self.xor_out, self.n)
        else:
            init = self.xor_out
        crc = _SoftwareCRC(self.n, self.n, self.poly, init, False, self.ref_out, 0)
        return crc._compute([0])

    @staticmethod
    def _reflect(word, n):
        """
        Bitwise-reflects an n-bit word `word`.
        """
        return int(f"{word:0{n}b}"[::-1], 2)

    def _generate_matrices(self):
        """
        Computes the F and G matrices for parallel CRC computation, treating
        the CRC as a linear time-invariant system described by the state
        relation x(t+1) = F.x(i) + G.u(i), where x(i) and u(i) are column
        vectors of the bits of the CRC register and input word, F is the n-by-n
        matrix relating the old state to the new state, and G is the n-by-m
        matrix relating the new data to the new state.

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

        These matrices are not affected by ``init``, ``ref_in``, ``ref_out``,
        or ``xor_out``.
        """
        f = []
        g = []
        for i in range(self.n):
            w = self._update(2 ** i, 0)
            f.append([int(x) for x in reversed(f"{w:0{self.n}b}")])
        for i in range(self.m):
            w = self._update(0, 2 ** i)
            g.append([int(x) for x in reversed(f"{w:0{self.n}b}")])
        return f, g


class CRC(_SoftwareCRC, Elaboratable):
    """Cyclic redundancy check (CRC) generator.

    This module generates CRCs from an input data stream, which can be used
    to validate an existing CRC or generate a new CRC. It can handle most
    forms of CRCs, with selectable polynomial, input data width, initial
    value, output XOR, and bit-reflection on input and output.

    It is parameterised using the well-known Williams model from
    "A Painless Guide to CRC Error Detection Algorithms":
    http://www.ross.net/crc/download/crc_v3.txt

    For the parameters to use for standard CRCs, refer to:

    * `reveng`_'s catalogue, which uses an identical parameterisation,
    * `crcmod`_'s list of predefined functions, but remove the leading '1'
      from the polynominal and where "Reversed" is True, set both ``ref_in``
      and ``ref_out`` to True,
    * `CRC Zoo`_, which contains only polynomials; use the "explicit +1"
      form of polynomial but remove the leading '1'.

    .. _reveng: https://reveng.sourceforge.io/crc-catalogue/all.htm
    .. _crcmod: http://crcmod.sourceforge.net/crcmod.predefined.html
    .. _CRC Zoo: https://users.ece.cmu.edu/~koopman/crc/

    The CRC value is updated on any clock cycle where ``valid`` is asserted,
    with the updated value available on the ``crc`` output on the subsequent
    clock cycle. The latency is therefore one cycle, and the throughput is
    one data word per clock cycle.

    With ``m=1``, a classic bit-serial CRC is implemented for the given
    polynomial in a Galois-type shift register. For larger values of m,
    a similar architecture computes every new bit of the CRC in parallel.

    The ``match`` output may be used to validate data with a trailing CRC
    (also known as a codeword). If the most recently processed word(s) form
    the valid CRC of all the previous data since reset, the CRC register
    will always take on a fixed value known as the residue. The ``match``
    output indicates whether the CRC register currently contains this residue.

    Parameters
    ----------
    n : int
        Bit width of CRC word. Also known as "width" in the Williams model.
    m : int
        Bit width of data words.
    poly : int
        CRC polynomial to use, n bits long, without the implicit x**n term.
        Polynomial is always specified with the highest order terms in the
        most significant bit positions; use ``ref_in`` and ``ref_out`` to
        perform a least significant bit first computation.
    init : int
        Initial value of CRC register at reset. Most significant bit always
        corresponds to the highest order term in the CRC register.
    ref_in : bool
        If True, the input data words are bit-reflected, so that they are
        processed least significant bit first.
    ref_out : bool
        If True, the output CRC is bit-reflected, so the least-significant bit
        of the output is the highest-order bit of the CRC register.
        Note that this reflection is performed over the entire CRC register;
        for transmission you may want to treat the output as a little-endian
        multi-word value, so for example the reflected 16-bit output 0x4E4C
        would be transmitted as the two octets 0x4C 0x4E, each transmitted
        least significant bit first.
    xor_out : int
        The output CRC will be the CRC register XOR'd with this value, applied
        after any output bit-reflection.

    Attributes
    ----------
    rst : Signal(), in
        Assert to re-initialise the CRC to the initial value.
    data : Signal(m), in
        Data word to add to CRC when ``valid`` is asserted.
    valid : Signal(), in
        Assert when ``data`` is valid to add the data word to the CRC.
        Ignored when ``rst`` is asserted.
    crc : Signal(n), out
        Registered CRC output value, updated one clock cycle after ``valid``
        becomes asserted.
    match : Signal(), out
        Asserted if the current CRC value indicates a valid codeword has been
        received.
    """
    def __init__(self, n, m, poly, init=0, ref_in=False, ref_out=False, xor_out=0):
        # Initialise SoftwareCRC arguments.
        super().__init__(n, m, poly, init, ref_in, ref_out, xor_out)

        self.rst = Signal()
        self.data = Signal(self.m)
        self.valid = Signal()
        self.crc = Signal(self.n)
        self.match = Signal()

    def elaborate(self, platform):
        m = Module()

        crc_reg = Signal(self.n, reset=self.init)
        data_in = Signal(self.m)

        # Optionally bit-reflect input words.
        if self.ref_in:
            m.d.comb += data_in.eq(self.data[::-1])
        else:
            m.d.comb += data_in.eq(self.data)

        # Optionally bit-reflect and then XOR the output from the state.
        if self.ref_out:
            m.d.comb += self.crc.eq(crc_reg[::-1] ^ self.xor_out)
        else:
            m.d.comb += self.crc.eq(crc_reg ^ self.xor_out)

        # Compute CRC matrices and expected residue using the software model.
        f, g = self._generate_matrices()
        residue = self._residue()

        # Compute next CRC state.
        with m.If(self.rst):
            m.d.sync += crc_reg.eq(self.init)
        with m.Elif(self.valid):
            for i in range(self.n):
                bit = 0
                for j in range(self.n):
                    if f[j][i]:
                        bit ^= crc_reg[j]
                for j in range(self.m):
                    if g[j][i]:
                        bit ^= data_in[j]
                m.d.sync += crc_reg[i].eq(bit)

        # Check for residue match, indicating a valid codeword.
        if self.ref_out:
            m.d.comb += self.match.eq(crc_reg[::-1] == residue)
        else:
            m.d.comb += self.match.eq(crc_reg == residue)

        return m
