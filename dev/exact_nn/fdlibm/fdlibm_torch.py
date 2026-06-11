#!/usr/bin/env python3
# EXACT-NN — bitwise-faithful torch float64 ports of V8's Math.exp / Math.pow.
#
# Source algorithm: v8/src/base/ieee754.cc, functions `exp(double)` and
# `pow(double, double)`, at V8 tag 11.3.244.8 — the V8 bundled by node
# v20.20.2 (deps/v8/src/base/ieee754.cc at nodejs/node tag v20.20.2 is
# byte-identical; verified by diff). That code is Sun's fdlibm e_exp.c /
# e_pow.c with V8's modifications (notably the exp(1)==E special case).
#
#   https://raw.githubusercontent.com/v8/v8/11.3.244.8/src/base/ieee754.cc
#   https://raw.githubusercontent.com/nodejs/node/v20.20.2/deps/v8/src/base/ieee754.cc
#
# Faithfulness contract:
#   * every floating-point intermediate is plain IEEE-754 float64 arithmetic
#     (+, -, *, /, sqrt) executed in the SAME order as the C code -> each op
#     is correctly rounded and therefore bit-identical on any IEEE machine
#     (CPU and CUDA float64 alike; proven op-by-op in ../spike_f64);
#   * the HI/LO 32-bit word surgery (EXTRACT_WORDS / GET_HIGH_WORD /
#     SET_LOW_WORD / INSERT_WORDS) is emulated with int64 bit ops on
#     .view(torch.int64); int32 wraparound is reproduced where the C code
#     relies on it;
#   * all constants are constructed from the exact hex bit patterns given in
#     the source comments (asserted below where the source only gives a
#     decimal literal);
#   * branches are vectorized as torch.where chains with the same priority
#     order as the C early returns. FULL branch coverage: no input class is
#     unsupported (NaN payloads, +-inf, +-0, subnormal args, subnormal
#     results, overflow/underflow boundaries, negative bases, huge |y|).
#
# NaN bit semantics (node-on-x86-64 ground truth, measured):
#   * `return x + x` / `return x + y` on NaN inputs: SSE quiets the FIRST NaN
#     operand and preserves its payload+sign -> emulated as bits|QUIET_BIT
#     (CUDA arithmetic would canonicalize NaNs, hence explicit bit ops);
#   * pow: |x|==1 with y=+-inf executes `return y - y` = inf - inf -> the x86
#     "indefinite" QNaN 0xFFF8000000000000;
#   * pow: (x<0)**(non-integer y) returns
#     std::numeric_limits<double>::signaling_NaN() = 0x7FF4000000000000,
#     and node hands those exact bits to JS (measured).
#
# js_exp / js_pow: float64 in/out, fully vectorized (no python loops over
# elements), CPU + CUDA.
import struct

import torch

__all__ = ["js_exp", "js_pow"]


def _hexf(u64):
    """Python float carrying exactly the given IEEE-754 bit pattern."""
    return struct.unpack("<d", struct.pack("<Q", u64))[0]


def _fbits(v):
    return struct.unpack("<Q", struct.pack("<d", v))[0]


# ---------------------------------------------------------------- constants
# exp() — hex pairs from ieee754.cc comments (== fdlibm e_exp.c)
_O_THRESHOLD = _hexf(0x40862E42FEFA39EF)  # 7.09782712893383973096e+02
_U_THRESHOLD = _hexf(0xC0874910D52D3051)  # -7.45133219101941108420e+02
_LN2HI = _hexf(0x3FE62E42FEE00000)        # ln2HI[0]; ln2HI[1] = -ln2HI[0]
_LN2LO = _hexf(0x3DEA39EF35793C76)        # ln2LO[0]; ln2LO[1] = -ln2LO[0]
_INVLN2 = _hexf(0x3FF71547652B82FE)
_P1 = _hexf(0x3FC555555555553E)
_P2 = _hexf(0xBF66C16C16BEBD93)
_P3 = _hexf(0x3F11566AAF25DE2C)
_P4 = _hexf(0xBEBBBD41C5D26BF1)
_P5 = _hexf(0x3E66376972BEA4D0)
_E_CONST = _hexf(0x4005BF0A8B145769)      # V8 special-cases exp(1.0) == E
_TWOM1000 = _hexf(0x0170000000000000)     # 2**-1000
_TWO1023 = _hexf(0x7FE0000000000000)      # 0x1p1023

# pow() — hex pairs from ieee754.cc comments (== fdlibm e_pow.c)
_DP_H1 = _hexf(0x3FE2B80340000000)
_DP_L1 = _hexf(0x3E4CFDEB43CFD006)
_TWO53 = _hexf(0x4340000000000000)
_L1 = _hexf(0x3FE3333333333303)
_L2 = _hexf(0x3FDB6DB6DB6FABFF)
_L3 = _hexf(0x3FD55555518F264D)
_L4 = _hexf(0x3FD17460A91D4101)
_L5 = _hexf(0x3FCD864A93C9DB65)
_L6 = _hexf(0x3FCA7E284A454EEF)
# pow shares P1..P5 with exp (same hex in both functions)
_LG2 = _hexf(0x3FE62E42FEFA39EF)
_LG2_H = _hexf(0x3FE62E4300000000)
_LG2_L = _hexf(0xBE205C610CA86C39)
_OVT = 8.0085662595372944372e-17   # decimal literal in the C source
_CP = _hexf(0x3FEEC709DC3A03FD)
_CP_H = _hexf(0x3FEEC709E0000000)
_CP_L = _hexf(0xBE3E2FE0145B01F5)
_IVLN2 = _hexf(0x3FF71547652B82FE)
_IVLN2_H = _hexf(0x3FF7154760000000)
_IVLN2_L = _hexf(0x3E54AE0BF85DDF44)
_THIRD = 0.3333333333333333333333  # C literal in the |y|-huge series

# scalbn (libm; operation is exactly defined: x*2^n correctly rounded.
# Implemented as fdlibm s_scalbn — any conforming scalbn is bit-identical.)
_TWO54 = _hexf(0x4350000000000000)
_TWOM54 = _hexf(0x3C90000000000000)

# correctly-rounded sqrt fixup (pow's y==0.5 path calls libm sqrt, which is
# IEEE correctly rounded; this torch build's CPU sqrt (MKL) is 1 ulp off on
# some inputs, so we re-round exactly ourselves)
_TWO106 = _hexf(0x4690000000000000)   # 2^106
_TWO108 = _hexf(0x46B0000000000000)   # 2^108
_DK_SPLIT = 134217729.0               # 2^27 + 1 (Dekker split)

_QUIET_BIT = 0x0008000000000000
_SNAN_BITS = 0x7FF4000000000000   # std::numeric_limits<double>::signaling_NaN()
_INDEF_BITS = 0xFFF8000000000000  # x86 result of inf - inf

# Constants only given as decimal literals in the C source: pin their bits.
assert _fbits(_OVT) == 0x3C971547652B82FE
assert _fbits(_THIRD) == 0x3FD5555555555555
assert _fbits(2.373) == 0x4002FBE76C8B4396   # policy exponent, for the record
assert _fbits(9.25) == 0x4022800000000000


# ------------------------------------------------------------------ helpers
def _bits(x):
    """float64 tensor -> int64 bit pattern."""
    return x.contiguous().view(torch.int64)


def _dbl(b):
    """int64 bit pattern -> float64 tensor."""
    return b.contiguous().view(torch.float64)


def _mk(hi, lo):
    """INSERT_WORDS: double from high/low 32-bit words (int64 tensors; any
    representation congruent mod 2^32 is accepted, mirroring C casts)."""
    hi = hi & 0xFFFFFFFF
    hi = hi - ((hi & 0x80000000) << 1)  # to signed int32 so << 32 cannot overflow
    return _dbl((hi << 32) | (lo & 0xFFFFFFFF))


def _setlow0(v):
    """SET_LOW_WORD(v, 0)."""
    return _dbl(_bits(v) & -0x100000000)


def _quiet(x):
    """x86 SSE NaN quieting: same bits with the quiet bit set."""
    return _dbl(_bits(x) | _QUIET_BIT)


def _cbits(u64, ref):
    """float64 scalar tensor with exact bits u64, on ref's device."""
    if u64 >= 1 << 63:
        u64 -= 1 << 64
    return torch.tensor(u64, dtype=torch.int64, device=ref.device).view(torch.float64)


def _wrap32(v):
    """int64 -> value of the same bits as int32 (C int truncation/wrap)."""
    return ((v + 0x80000000) & 0xFFFFFFFF) - 0x80000000


def _i64(val, ref):
    return torch.tensor(val, dtype=torch.int64, device=ref.device)


# ---------------------------------------------------------------------- exp
def js_exp(x: torch.Tensor) -> torch.Tensor:
    """Bitwise replica of v8::base::ieee754::exp (JS Math.exp in node)."""
    if x.dtype != torch.float64:
        raise TypeError("js_exp requires float64")
    x = x.contiguous()
    bits = _bits(x)
    hx_s = bits >> 32                  # signed high word
    xsb = (hx_s >> 31) & 1             # sign bit of x
    hx = hx_s & 0x7FFFFFFF             # high word of |x|
    lx = bits & 0xFFFFFFFF

    zero = torch.zeros_like(x)
    one = torch.ones_like(x)

    # --- special filter (C: early returns; priority NaN > inf > over > under)
    m_nan = (hx > 0x7FF00000) | ((hx == 0x7FF00000) & (lx != 0))
    m_inf = (hx == 0x7FF00000) & (lx == 0)
    m_of = (~m_nan) & (~m_inf) & (x > _O_THRESHOLD)   # -> huge*huge = +inf
    m_uf = (~m_nan) & (~m_inf) & (x < _U_THRESHOLD)   # -> twom1000^2 = +0
    m_early = m_nan | m_inf | m_of | m_uf

    # sanitize early lanes so the main path computes harmless garbage
    xs = torch.where(m_early, zero, x)
    hxs = torch.where(m_early, torch.zeros_like(hx), hx)

    # --- argument reduction
    m_big = hxs > 0x3FD62E42                  # |x| > 0.5 ln2
    m_mid = m_big & (hxs < 0x3FF0A2B2)        # |x| < 1.5 ln2
    m_far = m_big & (hxs >= 0x3FF0A2B2)
    m_small = hxs < 0x3E300000                # |x| < 2^-28 -> return 1+x
    m_is1 = m_mid & (xs == 1.0)               # V8: exp(1) == E exactly

    sgn = xsb == 1
    ln2hi_sel = torch.where(sgn, torch.full_like(x, -_LN2HI),
                            torch.full_like(x, _LN2HI))
    ln2lo_sel = torch.where(sgn, torch.full_like(x, -_LN2LO),
                            torch.full_like(x, _LN2LO))
    half_sel = torch.where(sgn, torch.full_like(x, -0.5),
                           torch.full_like(x, 0.5))

    # mid: k = 1 - xsb - xsb
    hi_mid = xs - ln2hi_sel
    lo_mid = ln2lo_sel
    k_mid = 1 - xsb - xsb

    # far: k = (int)(invln2*x + halF[xsb])  (C cast truncates toward zero)
    kf = torch.trunc(_INVLN2 * xs + half_sel)
    k_far = kf.to(torch.int64)
    t_far = kf                                 # t = k (double), exact
    hi_far = xs - t_far * _LN2HI               # t*ln2HI is exact here
    lo_far = t_far * _LN2LO

    k = torch.where(m_mid, k_mid, torch.where(m_far, k_far, torch.zeros_like(k_far)))
    hi = torch.where(m_mid, hi_mid, torch.where(m_far, hi_far, zero))
    lo = torch.where(m_mid, lo_mid, torch.where(m_far, lo_far, zero))
    xr = torch.where(m_big, hi - lo, xs)       # x now in primary range

    # --- core polynomial (exact transcription, same op order)
    t = xr * xr
    c = xr - t * (_P1 + t * (_P2 + t * (_P3 + t * (_P4 + t * _P5))))
    res_k0 = one - ((xr * c) / (c - 2.0) - xr)
    yv = one - ((lo - (xr * c) / (2.0 - c)) - hi)

    # twopk = 2^k via INSERT_WORDS (int32 wrap emulated by masking)
    hw = torch.where(k >= -1021,
                     (0x3FF00000 + ((k << 20) & 0xFFFFFFFF)) & 0xFFFFFFFF,
                     (0x3FF00000 + (((k + 1000) << 20) & 0xFFFFFFFF)) & 0xFFFFFFFF)
    twopk = _mk(hw, torch.zeros_like(hw))

    res = torch.where(
        k == 0, res_k0,
        torch.where(k == 1024, yv * 2.0 * _TWO1023,
                    torch.where(k >= -1021, yv * twopk, yv * twopk * _TWOM1000)))

    # --- early returns, lowest to highest priority
    res = torch.where(m_small, one + x, res)
    res = torch.where(m_is1, torch.full_like(x, _E_CONST), res)
    res = torch.where(m_uf, zero, res)
    res = torch.where(m_of, torch.full_like(x, float("inf")), res)
    res = torch.where(m_inf, torch.where(sgn, zero, x), res)
    res = torch.where(m_nan, _quiet(x), res)
    return res


# --------------------------------------------------------- correct sqrt
def _ieee_sqrt(x):
    """IEEE-754 correctly-rounded sqrt for x in [+0, +inf], any backend.

    libm/hardware sqrt is correctly rounded, but torch's CPU sqrt (MKL/SLEEF
    vector path) is occasionally 1 ulp off, so torch.sqrt is only used as a
    seed and the rounding decision is re-made EXACTLY:

      x = m * 2^E2 with E2 even, m in [1,4)  ->  sqrt(x) = sqrt(m) * 2^(E2/2).
      Around the seed r0 ~ sqrt(m), test the candidate midpoints w = a + h
      (a a double in [1,2), h = half the ulp gap). sign(w^2 - m) is computed
      with zero error: Dekker two-product splits w^2-m into exact float64
      terms that are all integer multiples of 2^-106 and < 2^63 * 2^-106 in
      magnitude, so they convert losslessly to int64 and the sign of their
      int64 sum is exact. Ties are impossible (a 53-bit double cannot equal
      the 106+-bit square of a midpoint). Negative/NaN lanes pass through
      (callers mask them); +-0 and +inf pass through per IEEE.
    """
    bits = _bits(x)
    m_zero = x == 0
    m_pass = (x < 0) | (x != x) | (bits == 0x7FF0000000000000) | m_zero
    xs = torch.where(m_pass, torch.ones_like(x), x)
    bs = _bits(xs)
    m_sub = bs < (1 << 52)                      # positive subnormal
    xs = torch.where(m_sub, xs * _TWO108, xs)   # exact 2^108 scaling
    eadj = torch.where(m_sub, torch.full_like(bs, -108), torch.zeros_like(bs))
    bs = _bits(xs)
    E = ((bs >> 52) - 1023) + eadj
    epar = E & 1
    e2 = E - epar                               # even part of the exponent
    m = _dbl(((1023 + epar) << 52) | (bs & 0xFFFFFFFFFFFFF))   # in [1,4)

    r0 = torch.sqrt(m)                          # faithful seed (<= 1 ulp off)
    rb = _bits(r0)
    count = torch.zeros_like(rb)
    one = torch.ones_like(m)
    for koff in (-3, -2, -1, 0, 1, 2):
        a = _dbl(rb + koff)
        b = _dbl(rb + (koff + 1))
        h = (b - a) * 0.5                       # exact power of two
        # Dekker two-product: a*a = p + e exactly
        t = a * _DK_SPLIT
        a_hi = t - (t - a)
        a_lo = a - a_hi
        p = a * a
        e = ((a_hi * a_hi - p) + a_hi * a_lo + a_hi * a_lo) + a_lo * a_lo
        d = p - m                               # Sterbenz: exact
        q2 = (a * h) * 2.0                      # exact (h is a power of two)
        tot = ((d * _TWO106).to(torch.int64) + (e * _TWO106).to(torch.int64)
               + (q2 * _TWO106).to(torch.int64) + (h * h * _TWO106).to(torch.int64))
        # m > w^2  <=>  w^2 - m < 0; out-of-band candidates decided a priori
        gt = torch.where(b <= 1.0, torch.ones_like(m_zero),
                         torch.where(a >= 2.0, torch.zeros_like(m_zero), tot < 0))
        count = count + gt.to(torch.int64)
    c = _dbl(rb + (count - 3))                  # correctly rounded sqrt(m)
    res = _dbl(_bits(c) + ((e2 >> 1) << 52))    # * 2^(E2/2), exact
    return torch.where(m_pass, x, res)


# ------------------------------------------------------------------- scalbn
def _scalbn(x, n):
    """fdlibm s_scalbn (libm scalbn is the exactly-rounded x*2^n; any
    conforming implementation returns identical bits). n: int64 tensor."""
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    inf = torch.full_like(x, float("inf"))

    bits = _bits(x)
    hx = bits >> 32
    lx = bits & 0xFFFFFFFF
    k0 = (hx & 0x7FF00000) >> 20               # extract exponent
    m_zero_exp = k0 == 0
    m_xzero = (lx | (hx & 0x7FFFFFFF)) == 0    # +-0
    m_subx = m_zero_exp & ~m_xzero

    xs = torch.where(m_subx, x * _TWO54, x)    # normalize subnormal x
    bits2 = _bits(xs)
    hx2 = bits2 >> 32
    lx2 = bits2 & 0xFFFFFFFF
    k = torch.where(m_subx, ((hx2 & 0x7FF00000) >> 20) - 54, k0)
    m_uf_tiny = m_subx & (n < -50000)          # tiny*x -> signed 0

    m_infnan = k == 0x7FF                      # return x + x
    kk = k + n
    m_ovf = kk > 0x7FE                         # huge*copysign(huge,x)
    m_norm = (kk > 0) & ~m_ovf
    z_norm = _mk((hx2 & 0x800FFFFF) | (kk << 20), lx2)
    m_under = kk <= -54
    m_under_ovf = m_under & (n > 50000)        # integer-overflow guard branch
    kk54 = kk + 54
    z_sub = _mk((hx2 & 0x800FFFFF) | ((kk54 & 0x7FF) << 20), lx2) * _TWOM54

    sgn = torch.where(hx < 0, -one, one)
    res = z_sub
    res = torch.where(m_under, torch.where(m_under_ovf, sgn * inf, sgn * zero), res)
    res = torch.where(m_norm, z_norm, res)
    res = torch.where(m_ovf, sgn * inf, res)
    res = torch.where(m_infnan, torch.where(x != x, _quiet(x), x), res)
    res = torch.where(m_uf_tiny, sgn * zero, res)
    res = torch.where(m_xzero, x, res)
    return res


# ---------------------------------------------------------------------- pow
def js_pow(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Bitwise replica of v8::base::ieee754::pow (JS Math.pow in node)."""
    if x.dtype != torch.float64 or y.dtype != torch.float64:
        raise TypeError("js_pow requires float64")
    x, y = torch.broadcast_tensors(x, y)
    x = x.contiguous()
    y = y.contiguous()

    bx = _bits(x)
    by = _bits(y)
    hx = bx >> 32                       # signed high words (C: int hx, hy)
    lx = bx & 0xFFFFFFFF                # unsigned low words
    hy = by >> 32
    ly = by & 0xFFFFFFFF
    ix = hx & 0x7FFFFFFF
    iy = hy & 0x7FFFFFFF

    zero = torch.zeros_like(x)
    one = torch.ones_like(x)
    inf = torch.full_like(x, float("inf"))
    i0 = torch.zeros_like(hx)
    snan = _cbits(_SNAN_BITS, x)
    indef = _cbits(_INDEF_BITS, x)

    # --- y == +-0: x**0 = 1 (checked before NaN)
    m_y0 = (iy | ly) == 0

    # --- NaN: return x + y (x86: quiet the first NaN operand)
    x_nan = (ix > 0x7FF00000) | ((ix == 0x7FF00000) & (lx != 0))
    y_nan = (iy > 0x7FF00000) | ((iy == 0x7FF00000) & (ly != 0))
    m_nan = x_nan | y_nan
    res_nan = torch.where(x_nan, _quiet(x), _quiet(y))

    # --- yisint: 0 non-integer, 1 odd, 2 even (only consulted when hx < 0)
    k_e = (iy >> 20) - 0x3FF
    sh1 = (52 - k_e).clamp(0, 63)              # k > 20 branch
    j1 = ly >> sh1
    yis1_ok = ((j1 << sh1) & 0xFFFFFFFF) == ly  # (j<<(52-k)) == (int)ly
    yis1 = torch.where(yis1_ok, 2 - (j1 & 1), i0)
    sh2 = (20 - k_e).clamp(0, 63)              # k <= 20 branch (needs ly==0)
    j2 = iy >> sh2
    yis2_ok = (ly == 0) & ((j2 << sh2) == iy)
    yis2 = torch.where(yis2_ok, 2 - (j2 & 1), i0)
    yisint = torch.where(
        hx < 0,
        torch.where(iy >= 0x43400000, torch.full_like(i0, 2),
                    torch.where(iy >= 0x3FF00000,
                                torch.where(k_e > 20, yis1, yis2), i0)),
        i0)

    # --- special y (ly == 0)
    m_yinf = (ly == 0) & (iy == 0x7FF00000)            # y = +-inf
    x_abs1 = ((ix - 0x3FF00000) | lx) == 0             # |x| == 1
    res_yinf = torch.where(
        x_abs1, indef,                                  # y - y = inf - inf
        torch.where(ix >= 0x3FF00000,
                    torch.where(hy >= 0, y, zero),
                    torch.where(hy < 0, -y, zero)))
    m_y1 = (ly == 0) & (iy == 0x3FF00000)              # y = +-1
    res_y1 = torch.where(hy < 0, one / x, x)
    m_y2 = (ly == 0) & (hy == 0x40000000)              # y = 2
    res_y2 = x * x
    m_yhalf = (ly == 0) & (hy == 0x3FE00000) & (hx >= 0)  # y = 0.5, x >= +0
    res_yhalf = _ieee_sqrt(x)                  # libm sqrt = correctly rounded

    ax = _dbl(bx & 0x7FFFFFFFFFFFFFFF)                 # fabs(x)

    # --- special x: +-0, +-inf, +-1 (lx == 0)
    m_xspec = (lx == 0) & ((ix == 0x7FF00000) | (ix == 0) | (ix == 0x3FF00000))
    z5 = ax
    z5 = torch.where(hy < 0, one / z5, z5)
    m_neg1_nonint = ((ix - 0x3FF00000) | yisint) == 0  # x==-1, y non-int
    z5 = torch.where(
        hx < 0,
        torch.where(m_neg1_nonint, snan,
                    torch.where(yisint == 1, -z5, z5)),
        z5)
    res_xspec = z5

    # --- (x<0)**(non-int) = sNaN
    n_sign = (hx >> 31) + 1                            # 0 if x<0 else 1
    m_negnan = (n_sign | yisint) == 0

    # s = -1 for (negative x)**(odd int), else +1
    s = torch.where((n_sign | (yisint - 1)) == 0, -one, one)

    # --- |y| huge
    m_yhuge = iy > 0x41E00000                          # |y| > 2^31
    m_yhuge2 = iy > 0x43F00000                         # |y| > 2^64
    res_yhuge2 = torch.where(
        ix <= 0x3FEFFFFF, torch.where(hy < 0, inf, zero),
        torch.where(hy > 0, inf, zero))
    m_xlow = ix < 0x3FEFFFFF
    m_xhigh = ix > 0x3FF00000
    res_yhuge_away = torch.where(
        m_xlow, torch.where(hy < 0, s * inf, s * zero),
        torch.where(hy > 0, s * inf, s * zero))
    res_yhuge = torch.where(m_yhuge2, res_yhuge2, res_yhuge_away)
    m_yhuge_ret = m_yhuge & (m_yhuge2 | m_xlow | m_xhigh)
    # series for |1-x| tiny (<= 2^-20): log(x) ~ x - x^2/2 + x^3/3 - x^4/4
    t_ser = ax - one                                   # t has 20 trailing zeros
    w_ser = (t_ser * t_ser) * (0.5 - t_ser * (_THIRD - t_ser * 0.25))
    u_ser = _IVLN2_H * t_ser                           # ivln2_h has 21 sig. bits
    v_ser = t_ser * _IVLN2_L - w_ser * _IVLN2
    t1_ser = _setlow0(u_ser + v_ser)
    t2_ser = v_ser - (t1_ser - u_ser)
    m_series = m_yhuge & ~m_yhuge_ret

    # --- main log2 path (computed on all lanes; garbage lanes masked later)
    m_subx = ix < 0x00100000                           # subnormal |x|
    ax9 = torch.where(m_subx, ax * _TWO53, ax)
    n9 = torch.where(m_subx, torch.full_like(i0, -53), i0)
    ix9 = (_bits(ax9) >> 32) & 0x7FFFFFFF
    n9 = n9 + ((ix9 >> 20) - 0x3FF)
    j9 = ix9 & 0x000FFFFF
    ixn = j9 | 0x3FF00000                              # normalize ix
    # determine interval: k=0 |x|<sqrt(3/2); k=1 |x|<sqrt(3); else round up
    k9 = torch.where(j9 <= 0x3988E, i0,
                     torch.where(j9 < 0xBB67A, torch.ones_like(i0), i0))
    m_up = j9 >= 0xBB67A
    n9 = n9 + torch.where(m_up, torch.ones_like(i0), i0)
    ixn = ixn - torch.where(m_up, torch.full_like(i0, 0x00100000), i0)
    ax9 = _mk(ixn, _bits(ax9) & 0xFFFFFFFF)            # SET_HIGH_WORD(ax, ix)
    bp_k = torch.where(k9 == 1, torch.full_like(x, 1.5), one)
    dp_h_k = torch.where(k9 == 1, torch.full_like(x, _DP_H1), zero)
    dp_l_k = torch.where(k9 == 1, torch.full_like(x, _DP_L1), zero)
    # ss = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5)
    u9 = ax9 - bp_k
    v9 = one / (ax9 + bp_k)
    ss = u9 * v9
    s_h = _setlow0(ss)
    t_h = _mk(((ixn >> 1) | 0x20000000) + 0x00080000 + (k9 << 18),
              torch.zeros_like(i0))                    # t_h = ax+bp[k] High
    t_l = ax9 - (t_h - bp_k)
    s_l = v9 * ((u9 - s_h * t_h) - s_h * t_l)
    # compute log(ax)
    s2 = ss * ss
    r9 = s2 * s2 * (_L1 + s2 * (_L2 + s2 * (_L3 + s2 * (_L4 + s2 * (_L5 + s2 * _L6)))))
    r9 = r9 + s_l * (s_h + ss)
    s2b = s_h * s_h
    t_h2 = _setlow0(3.0 + s2b + r9)
    t_l2 = r9 - ((t_h2 - 3.0) - s2b)
    # u+v = ss*(1+...)
    u9b = s_h * t_h2
    v9b = s_l * t_h2 + t_l2 * ss
    # 2/(3log2)*(ss+...)
    p_h9 = _setlow0(u9b + v9b)
    p_l9 = v9b - (p_h9 - u9b)
    z_h = _CP_H * p_h9
    z_l = _CP_L * p_h9 + p_l9 * _CP + dp_l_k
    # log2(ax) = (ss+..)*2/(3*log2) = n + dp_h + z_h + z_l
    t9 = n9.to(torch.float64)
    t1_main = _setlow0(((z_h + z_l) + dp_h_k) + t9)
    t2_main = z_l - (((t1_main - t9) - dp_h_k) - z_h)

    t1 = torch.where(m_series, t1_ser, t1_main)
    t2 = torch.where(m_series, t2_ser, t2_main)

    # --- split y = y1+y2, compute z = y*log2(x) in two pieces
    y1 = _setlow0(y)
    p_l = (y - y1) * t1 + y * t2
    p_h = y1 * t1
    z10 = p_l + p_h
    jz = _bits(z10) >> 32                              # signed (C: int j)
    iz = _bits(z10) & 0xFFFFFFFF                       # (C: int i)
    # overflow: z >= 1024
    m_of1 = jz >= 0x40900000
    m_of = m_of1 & (((jz != 0x40900000) | (iz != 0))
                    | ((p_l + _OVT) > (z10 - p_h)))
    # underflow: z <= -1075
    m_uf1 = (~m_of1) & ((jz & 0x7FFFFFFF) >= 0x4090CC00)
    m_uf = m_uf1 & ((((jz & 0xFFFFFFFF) != 0xC090CC00) | (iz != 0))
                    | (p_l <= (z10 - p_h)))

    # --- compute 2**(p_h+p_l)
    i11 = jz & 0x7FFFFFFF
    k11 = (i11 >> 20) - 0x3FF
    m_zbig = i11 > 0x3FE00000                          # |z| > 0.5 -> n = [z+0.5]
    c_100000 = _i64(0x00100000, x)
    c_fffff = _i64(0x000FFFFF, x)
    sh_a = (k11 + 1).clamp(0, 31)
    n11 = jz + (c_100000 >> sh_a)
    k11b = ((n11 & 0x7FFFFFFF) >> 20) - 0x3FF          # new k for n
    sh_b = k11b.clamp(0, 31)
    t11 = _mk(n11 & ~(c_fffff >> sh_b), torch.zeros_like(i0))
    sh_c = (20 - k11b).clamp(0, 63)
    n11v = ((n11 & 0x000FFFFF) | 0x00100000) >> sh_c
    n11v = torch.where(jz < 0, -n11v, n11v)
    n_fin = torch.where(m_zbig, n11v, i0)
    p_h = torch.where(m_zbig, p_h - t11, p_h)

    tt = _setlow0(p_l + p_h)
    u = tt * _LG2_H
    v = (p_l - (tt - p_h)) * _LG2 + tt * _LG2_L
    zz = u + v
    w = v - (zz - u)
    t = zz * zz
    t1f = zz - t * (_P1 + t * (_P2 + t * (_P3 + t * (_P4 + t * _P5))))
    r = (zz * t1f) / ((t1f - 2.0) - (w + zz * w))
    zz = one - (r - zz)
    j11 = _bits(zz) >> 32
    jj = _wrap32(j11 + ((n_fin << 20) & 0xFFFFFFFF))   # j += (int)((u32)n << 20)
    m_subres = (jj >> 20) <= 0
    z_highadd = _mk(jj, _bits(zz) & 0xFFFFFFFF)
    z_scalbn = _scalbn(zz, n_fin)                      # subnormal output
    z_fin = torch.where(m_subres, z_scalbn, z_highadd)
    res = s * z_fin

    # --- early returns, lowest to highest priority (mirrors C return order)
    res = torch.where(m_uf, s * zero, res)             # s*tiny*tiny
    res = torch.where(m_of, s * inf, res)              # s*huge*huge
    res = torch.where(m_yhuge_ret, res_yhuge, res)
    res = torch.where(m_negnan, snan, res)
    res = torch.where(m_xspec, res_xspec, res)
    res = torch.where(m_yhalf, res_yhalf, res)
    res = torch.where(m_y2, res_y2, res)
    res = torch.where(m_y1, res_y1, res)
    res = torch.where(m_yinf, res_yinf, res)
    res = torch.where(m_nan, res_nan, res)
    res = torch.where(m_y0, one, res)
    return res
