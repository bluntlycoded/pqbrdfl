"""ML-DSA_A: FIPS 204 ML-DSA with every SHAKE call routed through the
domain-separated Ascon sponge.

Built on the pure-Python reference ``dilithium-py`` (MIT, pinned in
requirements.txt).  The three internal algorithms (FIPS 204 Algorithms 6-8)
are restated here so that every hash call carries an explicit domain; the
lattice arithmetic, packing and rejection conditions are the library's own.
The four ring samplers are overridden in ``_AsconRing``.
"""

from __future__ import annotations

from dilithium_py.ml_dsa.default_parameters import DEFAULT_PARAMETERS
from dilithium_py.ml_dsa.ml_dsa import ML_DSA
from dilithium_py.polynomials.polynomials import PolynomialRing

from .domains import D, SP, SP_stream

_Q = 8380417


class _AsconRing(PolynomialRing):
    def sample_in_ball(self, seed, tau):
        xof = SP_stream(D.SIG_CHAL, seed)
        sign_int = int.from_bytes(xof.read(8), "little")
        coeffs = [0] * 256
        for i in range(256 - tau, 256):
            while True:
                j = xof.read(1)[0]
                if j <= i:
                    break
            coeffs[i] = coeffs[j]
            coeffs[j] = 1 - 2 * (sign_int & 1)
            sign_int >>= 1
        return self(coeffs)

    def rejection_sample_ntt_poly(self, rho, i, j):
        xof = SP_stream(D.SIG_XOF, rho + bytes([j, i]))
        coeffs = []
        while len(coeffs) < 256:
            v = int.from_bytes(xof.read(3), "little") & 0x7FFFFF
            if v < _Q:
                coeffs.append(v)
        return self(coeffs, is_ntt=True)

    def rejection_bounded_poly(self, rho_prime, i, eta):
        xof = SP_stream(D.SIG_PRF, rho_prime + int.to_bytes(i, 2, "little"))

        def half(j):
            if eta == 2 and j < 15:
                return 2 - (j % 5)
            if eta == 4 and j < 9:
                return 4 - j
            return None

        coeffs = []
        while len(coeffs) < 256:
            b = xof.read(1)[0]
            for c in (half(b % 16), half(b // 16)):
                if c is not None and len(coeffs) < 256:
                    coeffs.append(c)
        return self(coeffs)

    def sample_mask_polynomial(self, rho_prime, i, kappa, gamma_1):
        bit_count, total = (18, 576) if gamma_1 == (1 << 17) else (20, 640)
        seed = rho_prime + int.to_bytes(kappa + i, 2, "little")
        r = int.from_bytes(SP(D.SIG_Y, seed, total), "little")
        mask = (1 << bit_count) - 1
        return self([gamma_1 - ((r >> bit_count * k) & mask) for k in range(self.n)])


class ML_DSA_A(ML_DSA):
    def __init__(self, parameter_set: dict):
        super().__init__(parameter_set)
        ring = _AsconRing()
        self.M.ring = ring
        self.R = ring

    @staticmethod
    def _hd(dom: int, data: bytes, n: int) -> bytes:
        return SP(dom, data, n)

    def _keygen_internal(self, zeta: bytes):
        seed = self._hd(D.SIG_KG, zeta + bytes([self.k, self.l]), 128)
        rho, rho_prime, K = seed[:32], seed[32:96], seed[96:]
        A_hat = self._expand_matrix_from_seed(rho)
        s1, s2 = self._expand_vector_from_seed(rho_prime)
        t = (A_hat @ s1.to_ntt()).from_ntt() + s2
        t1, t0 = t.power_2_round(self.d)
        pk = self._pack_pk(rho, t1)
        tr = self._hd(D.SIG_TR, pk, 64)
        sk = self._pack_sk(rho, K, tr, s1, s2, t0)
        return pk, sk

    def _sign_internal(self, sk: bytes, m: bytes, rnd: bytes, external_mu: bool = False):
        rho, K, tr, s1, s2, t0 = self._unpack_sk(sk)
        s1_hat, s2_hat, t0_hat = s1.to_ntt(), s2.to_ntt(), t0.to_ntt()
        A_hat = self._expand_matrix_from_seed(rho)
        mu = m if external_mu else self._hd(D.SIG_MSG, tr + m, 64)
        rho_prime = self._hd(D.SIG_RHO, K + rnd + mu, 64)
        kappa = 0
        alpha = self.gamma_2 << 1
        while True:
            y = self._expand_mask_vector(rho_prime, kappa)
            w = (A_hat @ y.to_ntt()).from_ntt()
            kappa += self.l
            w1 = w.high_bits(alpha)
            c_tilde = self._hd(D.SIG_CHAL, mu + w1.bit_pack_w(self.gamma_2), self.c_tilde_bytes)
            c_hat = self.R.sample_in_ball(c_tilde, self.tau).to_ntt()
            z = y + s1_hat.scale(c_hat).from_ntt()
            if z.check_norm_bound(self.gamma_1 - self.beta):
                continue
            c_s2 = s2_hat.scale(c_hat).from_ntt()
            if (w - c_s2).low_bits(alpha).check_norm_bound(self.gamma_2 - self.beta):
                continue
            c_t0 = t0_hat.scale(c_hat).from_ntt()
            if c_t0.check_norm_bound(self.gamma_2):
                continue
            h = (-c_t0).make_hint(w - c_s2 + c_t0, alpha)
            if h.sum_hint() > self.omega:
                continue
            return self._pack_sig(c_tilde, z, h)

    def _verify_internal(self, pk: bytes, m: bytes, sig: bytes) -> bool:
        rho, t1 = self._unpack_pk(pk)
        try:
            c_tilde, z, h = self._unpack_sig(sig)
        except ValueError:
            return False
        if h.sum_hint() > self.omega or z.check_norm_bound(self.gamma_1 - self.beta):
            return False
        A_hat = self._expand_matrix_from_seed(rho)
        tr = self._hd(D.SIG_TR, pk, 64)
        mu = self._hd(D.SIG_MSG, tr + m, 64)
        c = self.R.sample_in_ball(c_tilde, self.tau).to_ntt()
        t1s = t1.scale(1 << self.d).to_ntt()
        w_approx = ((A_hat @ z.to_ntt()) - t1s.scale(c)).from_ntt()
        w1 = h.use_hint(w_approx, 2 * self.gamma_2)
        return c_tilde == self._hd(D.SIG_CHAL, mu + w1.bit_pack_w(self.gamma_2), self.c_tilde_bytes)


ML_DSA_A_44 = ML_DSA_A(DEFAULT_PARAMETERS["ML_DSA_44"])
ML_DSA_A_65 = ML_DSA_A(DEFAULT_PARAMETERS["ML_DSA_65"])
