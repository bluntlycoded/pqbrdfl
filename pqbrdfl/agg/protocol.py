"""Layer 3: committee-based, secret-shared, Byzantine-robust aggregation.

One round (manuscript Appendix B, Algorithm 1):

  0. Sortition.  Committee C_r (m >= 4t+1 agents in exact-norm mode,
     m >= 3t+1 in sketch mode) = the m smallest
     H(SORT, beacon_{r-1} || r || id).  Anyone can recompute it.
  1. Keys.  Each member publishes a signed ephemeral KEM key for round r.
  2. Share.  Agent i encodes y_i = x_i - v (v: public centre, the previous
     aggregate), Shamir-shares it with threshold t, and seals share j to
     member j over the PQ channel (Layer 1).
  3. Commit.  Each member opens its shares and signs a digest of what it
     received.  The digests fix the challenge seed
         s_r = H(AGG_CHAL, r || beacon || commitments),
     which expands to k random 0/1 vectors R and a field vector rho.
  4. Validate.  Each member publishes, per client, shares of
         R y_i (k values, degree t),  1^T y_i (degree t),  ||y_i||^2 (degree 2t).
     Everyone decodes them (Reed-Solomon, errors corrected/identified) and
     rejects a client if any projection leaves [-T, T]: this forces every
     coordinate into [-2T, 2T] except with probability 2^-k, which in turn
     makes the decoded ||y_i||^2 the true integer norm (no modular wrap).
  5. Clip.  Centred clipping weights w_i = min(1, tau / ||y_i||) / n_acc are
     public (they depend only on the published norms).
  6. Aggregate.  Members publish shares of sum_i w_i y_i; everyone decodes
     it with error identification via the rho projection, and sets
         v_{r+1} = v + sum_i w_i y_i.
  7. Certify.  The round certificate is appended to the Merkle audit log and
     becomes the next beacon.

Leakage (stated, not hidden): for every accepted client the public learns
||y_i||, R y_i and 1^T y_i -- k + 2 scalars of a d-dimensional vector (sketch
mode: only R y_i and 1^T y_i).  Opened degree-t polynomials are determined by
the t corrupt shares plus their constant term, so they reveal nothing more;
the degree-2t norm polynomials are re-randomised with a structural zero
sharing before opening, for the same reason.
Everything else about y_i is information-theoretically hidden from any
coalition of at most t members, and the transport is post-quantum.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field as dfield

import numpy as np

from ..crypto import channel as ch
from ..crypto.backend import Backend, get_backend
from ..crypto.domains import D
from ..crypto.merkle import MerkleLog
from . import field as F
from . import shamir as S
from .committee import select_committee

HONEST = "honest"
# client (sender) misbehaviour at the protocol level
C_WRAP = "wrap"                  # modular wrap-around vector (passes a naive norm check)
C_INCONSISTENT = "inconsistent"  # shares not on one degree-t polynomial
C_FORGE = "forge"                # message signed with an unregistered key
C_RAW_HUGE = "huge"              # honest encoding of a huge vector
# committee-member misbehaviour
M_CORRUPT_SCALARS = "corrupt_scalars"
M_CORRUPT_AGG = "corrupt_agg"
M_COLLUDE = "collude"            # passive: pools its shares with the adversary
M_ORTHO = "ortho"                # adaptive: aggregate error orthogonal to the pre-published rho


@dataclass
class AggConfig:
    m: int = 7
    t: int = 2
    frac_bits: int = 14
    w_frac_bits: int = 20
    k_proj: int = 64
    range_check: bool = True
    # "exact": ||y||^2 from degree-2t product shares; guaranteed output delivery
    #          against <= t corrupt members needs m >= 4t + 1.
    # "sketch": ||y||^2 estimated from the k published projections
    #          (Rademacher / JL estimate, relative s.d. ~ sqrt(2/k)); degree t
    #          only, so m >= 3t + 1 suffices.  Publishes nothing beyond R y, 1^T y.
    norm_mode: str = "exact"
    robust: str = "cc"            # "cc" | "mkcc" (sketch multi-Krum + CC) | "mean"
    mk_q: float = 0.5             # mkcc: fraction of validated clients selected
    krum_f: int | None = None     # mkcc: assumed Byzantine count (default floor(n/4))
    # "correct": Reed-Solomon error correction, guaranteed output while corrupt
    #            members stay within the decoding radius (needs m >= 3t+1 / 4t+1).
    # "abort":   accept a value only if all m shares agree (security with abort).
    #            Privacy needs c <= t, safety needs c <= m - deg - 1, so with
    #            t = (m-1)/2 an honest-majority committee suffices; a failed
    #            round is retried with a fresh committee that excludes suspects.
    decode_policy: str = "correct"
    excl_frac: float = 0.1        # abort mode: max fraction of clients rejected as inconsistent
    max_retries: int = 3
    tau_mult: float = 1.0         # tau = tau_mult * median accepted norm
    backend: str = "fips"

    def __post_init__(self):
        deg_mult = 2 if self.norm_mode == "exact" else 1
        need = (deg_mult * self.t + self.t + 1) if self.decode_policy == "abort" else 3 * self.t + 1
        if self.m < need:
            raise ValueError(f"need m >= {need} for policy {self.decode_policy}/{self.norm_mode}")
        if self.norm_mode not in ("exact", "sketch"):
            raise ValueError(self.norm_mode)


@dataclass
class RoundResult:
    round: int
    committee: list[str]
    aggregate: np.ndarray | None
    accepted: list[str]
    rejected: dict[str, str]
    norms: dict[str, float]
    weights: dict[str, float]
    flagged_members: set[str]
    captured: bool
    privacy_breach: bool
    recovered_updates: dict[str, np.ndarray]
    bytes_by_kind: dict[str, int]
    time_by_phase: dict[str, float]
    certificate: bytes = b""
    aborted: bool = False
    banned: int = 0
    suspects: set = dfield(default_factory=set)
    attempts: int = 1


def range_bound(d: int, frac_bits: int) -> int:
    """Largest T (field units) with d * (2T)^2 < p/2, i.e. no wrap in ||y||^2."""
    return math.isqrt((F.P // 2) // d) // 2


def _expand(backend: Backend, seed: bytes, n_bytes: int) -> bytes:
    if backend.name == "ascon":
        from ..crypto.domains import SP_stream
        return SP_stream(D.AGG_CHAL, seed).read(n_bytes)
    return hashlib.shake_256(bytes([int(D.AGG_CHAL)]) + seed).digest(n_bytes)


def challenge(backend: Backend, seed: bytes, k: int, d: int) -> tuple[np.ndarray, np.ndarray]:
    raw = _expand(backend, seed, (k * d + 7) // 8 + 8 * d)
    bits = np.unpackbits(np.frombuffer(raw[: (k * d + 7) // 8], np.uint8))[: k * d]
    R = bits.reshape(k, d).astype(np.uint8)
    rho = np.frombuffer(raw[(k * d + 7) // 8:], np.uint64)[:d] & F.P_U
    return R, rho


class AggregationNetwork:
    """All agents' identities, the public registry, the beacon and the audit log."""

    def __init__(self, agent_ids: list[str], cfg: AggConfig):
        self.cfg = cfg
        self.backend = get_backend(cfg.backend)
        self.ids = list(agent_ids)
        self.identity = {a: ch.Identity.generate(self.backend, a) for a in self.ids}
        self.registry = {a: self.identity[a].pk for a in self.ids}
        self.beacon = self.backend.H(D.ROUND, b"genesis", 32)
        self.audit = MerkleLog(self.backend.H)
        # Members caught publishing a value that disagrees with the decoded codeword.
        # The evidence is their own signed output, which anyone can recheck.  Capped
        # at N/4 so that framing (possible only beyond the decoding radius) cannot
        # shrink the eligible pool without bound.
        self.banned: set[str] = set()
        self.ban_cap = len(self.ids) // 4

    # ------------------------------------------------------------ validation
    def _accountable_validation(self, R, included, dealt, held, committee, rows_proj, rows_s1,
                                xs, t, n, rejected, proven_bad, nbytes, B, rnd):
        """Validate projections against the clients' signed claims; settle disputes.

        Returns decoded projection / sum values (lists indexed like batch_decode)."""
        m, k = len(committee), R.shape[0]
        # 1. clients publish claims pi_ij = (R s_ij, 1^T s_ij)
        C = np.zeros((m, (k + 1) * n), np.uint64)
        for col, i in enumerate(included):
            sh = dealt[i]                                      # (m, d): what client i dealt
            proj = F.dot_binary(R, sh.T)                       # (k, m)
            s1 = F.sum_mod(sh, axis=1)                         # (m,)
            for j in range(m):
                C[j, [l * n + col for l in range(k)]] = proj[:, j]
                C[j, k * n + col] = s1[j]
            payload = proj.tobytes() + s1.tobytes()
            nbytes["claims"] += len(payload) + len(ch.sign_statement(B, self.identity[i], "claims", rnd,
                                                                     B.H(D.ROUND, payload, 32)))
        # 2. a dealing is valid iff every claimed column lies on one degree-t polynomial
        vals, _ = S.batch_decode(xs, C, t, 0)
        for col, i in enumerate(included):
            if any(vals[l * n + col] is None for l in range(k + 1)) and i not in rejected:
                rejected[i] = "inconsistent dealing"
        # 3. member publications must equal the claims; a mismatch is a dispute: the
        #    member reveals the channel key K_ij, anyone decrypts client i's signed
        #    ciphertext and recomputes the projection of the real share.
        Pm = np.stack(rows_proj)                               # (m, k*n)
        Sm = np.stack(rows_s1)                                 # (m, n)
        for col, i in enumerate(included):
            if i in rejected:
                continue
            cols = [l * n + col for l in range(k)]
            for j in range(m):
                if np.array_equal(Pm[j, cols], C[j, cols]) and Sm[j, col] == C[j, k * n + col]:
                    continue
                real = held[committee[j]][i]                   # = AEAD-authenticated share
                truth = np.concatenate([F.dot_binary(R, real), [F.sum_mod(real)]])
                claim = np.concatenate([C[j, cols], [C[j, k * n + col]]])
                nbytes["disputes"] += real.nbytes + 32
                if np.array_equal(truth, claim):
                    proven_bad.add(j)                          # member published a false value
                else:
                    rejected[i] = "false claim"                # client lied about its dealing
                    break
        proj_vals = [vals[c] for c in range(k * n)]
        s1_vals = [vals[k * n + c] for c in range(n)]
        return proj_vals, s1_vals

    # ------------------------------------------------------------------ round
    def run_round(self, rnd: int, updates: dict[str, np.ndarray], center: np.ndarray,
                  client_behavior: dict[str, str] | None = None,
                  member_behavior: dict[str, str] | None = None,
                  rng: np.random.Generator | None = None,
                  attempt: int = 0, exclude: set | None = None) -> RoundResult:
        cfg, B = self.cfg, self.backend
        exclude = exclude or set()
        maxe = 0 if cfg.decode_policy == "abort" else None
        rng = rng or np.random.default_rng()
        client_behavior = client_behavior or {}
        member_behavior = member_behavior or {}
        H = B.H
        nbytes: dict[str, int] = defaultdict(int)
        tphase: dict[str, float] = defaultdict(float)
        d = center.shape[0]
        m, t, Fb = cfg.m, cfg.t, cfg.frac_bits
        xs = list(range(1, m + 1))

        # 0. sortition --------------------------------------------------------
        seed_beacon = self.beacon if attempt == 0 else H(D.ROUND, ch.enc(self.beacon, "retry", attempt), 32)
        committee = select_committee(H, seed_beacon, rnd, [a for a in self.ids if a not in exclude], m)
        if len(committee) < m:
            # not enough eligible agents left (bans / exclusions): run with the smaller
            # committee only if it still meets the policy minimum for threshold t
            m = len(committee)
            xs = list(range(1, m + 1))
            need = (2 * t + 1) if cfg.decode_policy == "abort" else (3 * t + 1)
            if m < need:
                return RoundResult(round=rnd, committee=committee, aggregate=None, accepted=[],
                                   rejected={i: "no committee" for i in updates}, norms={}, weights={},
                                   flagged_members=set(), captured=False, privacy_breach=False,
                                   recovered_updates={}, bytes_by_kind={}, time_by_phase={},
                                   aborted=True, suspects=set(), attempts=attempt + 1)
        corrupt = [c for c in committee if member_behavior.get(c, HONEST) != HONEST]
        captured = len(corrupt) > t

        # 1. ephemeral keys ---------------------------------------------------
        t0 = time.perf_counter()
        eph = {c: ch.EphemeralKey.generate(B, self.identity[c], rnd) for c in committee}
        for c in committee:
            nbytes["epk"] += len(eph[c].pk) + len(eph[c].sig)
        tphase["member_keys"] += time.perf_counter() - t0

        # 2. clients share and seal ------------------------------------------
        t0 = time.perf_counter()
        inbox: dict[str, list[ch.Sealed]] = {c: [] for c in committee}
        dealt: dict[str, np.ndarray] = {}
        epk_ok_cache: dict[str, bool] = {}
        for c in committee:
            e = eph[c]
            epk_ok_cache[c] = ch.verify_ephemeral(B, self.registry[c], c, rnd, e.pk, e.sig)
        for i, x in updates.items():
            beh = client_behavior.get(i, HONEST)
            y = np.asarray(x, np.float64) - center
            if beh == C_RAW_HUGE:
                y = y * 1e6
            try:
                yf = F.encode(y, Fb, rng)
            except OverflowError:
                continue
            if beh == C_WRAP:
                yf = yf.copy()
                idx = rng.choice(d, size=4, replace=False)
                yf[idx] = np.uint64((F.P + 1) // 2)
            sh = S.share(yf, t, m)
            if beh == C_INCONSISTENT:
                sh[0] = F.add(sh[0], F.rand(d))
            dealt[i] = sh
            sender = self.identity[i]
            if beh == C_FORGE:
                sender = ch.Identity.generate(B, i)       # not the registered key
            for j, c in enumerate(committee):
                if not epk_ok_cache[c]:
                    continue
                msg = ch.seal(B, sender, c, eph[c].pk, rnd, "share", sh[j].tobytes())
                inbox[c].append(msg)
                nbytes["shares"] += msg.wire_size()
        tphase["client_share_seal"] += time.perf_counter() - t0

        # 3. members open and commit -----------------------------------------
        t0 = time.perf_counter()
        held: dict[str, dict[str, np.ndarray]] = {c: {} for c in committee}
        for c in committee:
            for msg in inbox[c]:
                if msg.sender not in self.registry:
                    continue
                try:
                    pt = ch.open_sealed(B, msg, self.registry[msg.sender], eph[c])
                except ch.ChannelError:
                    continue
                v = np.frombuffer(pt, np.uint64)
                if v.shape[0] != d or np.any(v >= F.P_U):
                    continue
                held[c][msg.sender] = v
        tphase["member_open"] += time.perf_counter() - t0

        rejected: dict[str, str] = {}
        included = [i for i in updates if all(i in held[c] for c in committee)]
        for i in updates:
            if i not in included:
                rejected[i] = "channel"          # bad signature / not delivered to all members
        commits = []
        for c in committee:
            dig = H(D.ROUND, ch.enc(rnd, c, *included), 32)
            sig = ch.sign_statement(B, self.identity[c], "commit", rnd, dig)
            commits.append(dig + sig)
            nbytes["commit"] += len(dig) + len(sig)
        seed = H(D.AGG_CHAL, ch.enc(rnd, self.beacon, *commits), 32)
        R, rho = challenge(B, seed, cfg.k_proj, d)
        # Accountable validation (abort policy, sketch norms): once R is fixed, every
        # client publishes signed projections of the share it dealt to each member,
        # pi_ij = (R s_ij, 1^T s_ij).  These reveal nothing beyond R y_i and 1^T y_i
        # (the projected sharing polynomial has independent uniform coefficients), let
        # anyone check the dealing, and turn every member/client mismatch into a
        # dispute settled by the member revealing that one channel key.
        use_claims = maxe is not None and cfg.norm_mode == "sketch"
        claims: dict[str, np.ndarray] = {}
        proven_bad: set[int] = set()

        # privacy accounting: colluding members pool their shares
        colluders = [j for j, c in enumerate(committee) if member_behavior.get(c) == M_COLLUDE
                     or member_behavior.get(c) in (M_CORRUPT_AGG, M_CORRUPT_SCALARS)]
        recovered: dict[str, np.ndarray] = {}
        privacy_breach = len(colluders) > t
        if privacy_breach:
            use = colluders[: t + 1]
            for i in included:
                shares_i = np.stack([held[committee[j]][i] for j in use])
                recovered[i] = F.decode(S.reconstruct([xs[j] for j in use], shares_i), Fb) + center

        # 4. validation scalars ----------------------------------------------
        t0 = time.perf_counter()
        n = len(included)
        rows_proj, rows_s1, rows_n2 = [], [], []
        zero_shares = None
        if cfg.norm_mode == "exact" and n:
            # Re-randomise the degree-2t norm sharings before opening (BGW).  Member j
            # sends member k the value r_j(x_k), deg r_j <= 2t-1; member k adds
            # x_k * r_j(x_k).  The sum over j is a degree-2t sharing whose constant
            # term is 0 *by construction*, so a Byzantine dealer cannot shift the
            # norms; it can only raise the degree, which decoding detects.  With one
            # honest dealer the opened polynomial is uniform given ||y||^2.
            zero_shares = np.zeros((m, n), np.uint64)
            for j, c in enumerate(committee):
                coeffs = F.rand((2 * t, n))
                for k in range(m):
                    x = np.uint64(xs[k])
                    val = np.zeros(n, np.uint64)
                    for cf in coeffs[::-1]:
                        val = F.add(F.mul(val, x), cf)
                    if k != j:
                        msg = ch.seal(B, self.identity[c], committee[k], eph[committee[k]].pk, rnd,
                                      "zero", val.tobytes())
                        nbytes["zero_sharing"] += msg.wire_size()
                        val = np.frombuffer(ch.open_sealed(B, msg, self.registry[c], eph[committee[k]]),
                                            np.uint64)
                    zero_shares[k] = F.add(zero_shares[k], F.mul(val, x))
        for j, c in enumerate(committee):
            if n == 0:
                break
            Y = np.stack([held[c][i] for i in included])          # (n, d)
            proj = F.dot_binary(R, Y.T)                            # (k, n)
            s1 = F.sum_mod(Y, axis=1)                              # (n,)
            n2 = (F.add(F.sum_mod(F.mul(Y, Y), axis=1), zero_shares[j]) if cfg.norm_mode == "exact"
                  else np.zeros(n, np.uint64))                     # (n,)
            if member_behavior.get(c) == M_CORRUPT_SCALARS:
                proj = F.add(proj, F.rand(proj.shape))
                n2 = F.add(n2, F.rand(n2.shape))
            payload = proj.tobytes() + s1.tobytes() + n2.tobytes()
            sig = ch.sign_statement(B, self.identity[c], "scalars", rnd, H(D.ROUND, payload, 32))
            nbytes["member_scalars"] += len(payload) + len(sig)
            rows_proj.append(proj.reshape(-1))
            rows_s1.append(s1)
            rows_n2.append(n2)
        tphase["member_validate"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        flagged: set[int] = set()
        norms: dict[str, float] = {}
        T = range_bound(d, Fb)
        accepted: list[str] = []
        if n:
            if use_claims:
                proj_vals, s1_vals = self._accountable_validation(
                    R, included, dealt, held, committee, rows_proj, rows_s1, xs, t, n,
                    rejected, proven_bad, nbytes, B, rnd)
                flagged |= proven_bad
            else:
                proj_vals, bad_p = S.batch_decode(xs, np.stack(rows_proj), t, maxe)
                s1_vals, bad_s = S.batch_decode(xs, np.stack(rows_s1), t, maxe)
                flagged |= bad_p | bad_s
            if cfg.norm_mode == "exact":
                N2 = np.stack(rows_n2)
                n2_vals, bad_n = S.batch_decode(xs, N2, 2 * t, maxe)
                flagged |= bad_n
                # erasure-assisted retry: drop members already identified in the
                # degree-t decodings and decode the degree-2t values again
                if any(v is None for v in n2_vals) and flagged and maxe is None:
                    keep = [j for j in range(m) if j not in flagged]
                    for col, v in enumerate(n2_vals):
                        if v is None and len(keep) >= 2 * t + 1:
                            r2 = S.decode_scalar([xs[j] for j in keep], [int(N2[j, col]) for j in keep], 2 * t)
                            n2_vals[col] = None if r2 is None else r2[0]
            k = cfg.k_proj
            for col, i in enumerate(included):
                if i in rejected:
                    continue
                pv = [proj_vals[l * n + col] for l in range(k)]
                if any(v is None for v in pv) or s1_vals[col] is None:
                    rejected[i] = "undecodable"
                    continue
                if cfg.range_check and any(abs(F.centred(v)) > T for v in pv):
                    rejected[i] = "range"
                    continue
                if cfg.norm_mode == "exact":
                    nv = n2_vals[col]
                    if nv is None:
                        rejected[i] = "undecodable"
                        continue
                    n2c = F.centred(nv)
                    if n2c < 0:
                        rejected[i] = "norm"
                        continue
                else:
                    tot = F.centred(s1_vals[col])
                    n2c = sum((2 * F.centred(v) - tot) ** 2 for v in pv) / k
                norms[i] = math.sqrt(n2c) / (1 << Fb)
                accepted.append(i)
        tphase["public_decode"] += time.perf_counter() - t0

        def _abort():
            return RoundResult(
                round=rnd, committee=committee, aggregate=None, accepted=[], rejected=rejected,
                norms=norms, weights={}, flagged_members={committee[j] for j in flagged},
                captured=captured, privacy_breach=privacy_breach, recovered_updates=recovered,
                bytes_by_kind=dict(nbytes), time_by_phase=dict(tphase), aborted=True,
                suspects={committee[j] for j in flagged}, attempts=attempt + 1)

        n_incons = sum(1 for r in rejected.values() if r == "undecodable")
        if maxe is not None and not use_claims and n and n_incons > cfg.excl_frac * n:
            return _abort()     # too many inconsistent columns: members are misbehaving

        # 5. clipping weights -------------------------------------------------
        weights: dict[str, float] = {}
        if accepted:
            pool = list(accepted)
            if cfg.robust == "mkcc" and len(accepted) > 2:
                # Multi-Krum on public Rademacher sketches S_i = 2 R y_i - 1^T y_i (the
                # range-check projections, already published): no additional leakage.
                from .robust import krum_select
                colof = {i: included.index(i) for i in accepted}
                Sk = np.array([[2 * F.centred(proj_vals[l * n + colof[i]]) - F.centred(s1_vals[colof[i]])
                                for l in range(cfg.k_proj)] for i in accepted], dtype=np.float64) / (1 << Fb)
                G = Sk @ Sk.T
                dg = np.diag(G)
                D2 = np.maximum(dg[:, None] + dg[None, :] - 2 * G, 0.0) / cfg.k_proj
                f_ass = cfg.krum_f if cfg.krum_f is not None else len(accepted) // 4
                q = max(1, int(round(cfg.mk_q * len(accepted))))
                pool = [accepted[j] for j in sorted(krum_select(D2, f_ass, q))]
            if cfg.robust in ("cc", "mkcc"):
                tau = cfg.tau_mult * float(np.median([norms[i] for i in pool]))
                for i in pool:
                    weights[i] = min(1.0, tau / norms[i]) if norms[i] > 0 else 1.0
            else:
                weights = {i: 1.0 for i in accepted}
            na = len(weights)
            weights = {i: w / na for i, w in weights.items()}
            accepted_all = accepted
            accepted = [i for i in accepted if i in weights]

        # 6. aggregate --------------------------------------------------------
        t0 = time.perf_counter()
        aggregate = None
        if accepted:
            Wq = {i: np.uint64(F.encode_scalar(weights[i], cfg.w_frac_bits)) for i in accepted}
            A_rows = []
            for c in committee:
                acc = np.zeros(d, np.uint64)
                for i in accepted:
                    acc = F.add(acc, F.mul(Wq[i], held[c][i]))
                if member_behavior.get(c) == M_CORRUPT_AGG:
                    acc = F.add(acc, F.rand(d))
                if member_behavior.get(c) == M_ORTHO:
                    # e with <e, rho> = 0 mod p: e_0 = rho_1, e_1 = -rho_0
                    e = np.zeros(d, np.uint64)
                    e[0], e[1] = rho[1], F.neg(np.array([rho[0]], np.uint64))[0]
                    acc = F.add(acc, e)
                sig = ch.sign_statement(B, self.identity[c], "aggregate", rnd, H(D.ROUND, acc.tobytes(), 32))
                nbytes["member_agg"] += acc.nbytes + len(sig)
                A_rows.append(acc)
            Arr = np.stack(A_rows)
            # commit-then-reveal: members first sign H(A_j); the identification vector
            # rho_A is derived from those commitments, so no member can choose an error
            # orthogonal to it.  (A rho fixed before A_j is published would let a
            # corrupt member add e with <e, rho> = 0 and pass the projection check.)
            a_commits = [H(D.ROUND, ch.enc("agg-commit", rnd, j, Arr[j].tobytes()), 32) for j in range(m)]
            nbytes["member_agg"] += 32 * m
            _, rho_a = challenge(B, H(D.AGG_CHAL, ch.enc("agg", rnd, seed, *a_commits), 32), 1, d)
            live = [j for j in range(m) if j not in proven_bad]
            xs_l = [xs[j] for j in live]
            if maxe is not None:
                # abort policy: accept only if every remaining member's vector lies on
                # one degree-t polynomial (checked on all d coordinates, not a projection)
                ok = len(live) >= t + 1 and S.consistent_rows(xs_l, Arr[live], t)
                if not ok:
                    proj_a = F.sum_mod(F.mul(Arr[live], rho_a[None, :]), axis=1)
                    r_a = S.decode_scalar(xs_l, [int(v) for v in proj_a], t)
                    if r_a is not None:
                        flagged |= {live[j] for j in r_a[1]}
                    tphase["aggregate"] += time.perf_counter() - t0
                    return _abort()
                A = S.reconstruct(xs_l[:t + 1], Arr[live[:t + 1]])
                dec = (A, [])
            else:
                dec = S.decode_vector(xs_l, Arr[live], t, rho_a, None)
                if dec is not None:
                    dec = (dec[0], [live[j] for j in dec[1]])
            if dec is not None:
                A, bad = dec
                flagged |= set(bad)
                aggregate = center + F.decode(A, Fb + cfg.w_frac_bits)
        tphase["aggregate"] += time.perf_counter() - t0

        # 7. certificate ------------------------------------------------------
        agg_digest = H(D.ROUND, b"" if aggregate is None else aggregate.astype(np.float64).tobytes(), 32)
        cert = H(D.ROUND, ch.enc(rnd, self.beacon, *committee, *accepted, agg_digest), 32)
        self.audit.append(cert)
        self.beacon = cert

        return RoundResult(
            round=rnd, committee=committee, aggregate=aggregate, accepted=accepted,
            rejected=rejected, norms=norms, weights=weights,
            flagged_members={committee[j] for j in flagged}, captured=captured,
            privacy_breach=privacy_breach, recovered_updates=recovered,
            bytes_by_kind=dict(nbytes), time_by_phase=dict(tphase), certificate=cert,
            suspects={committee[j] for j in proven_bad})


def run_round_with_retries(net: "AggregationNetwork", rnd: int, updates, center, **kw) -> RoundResult:
    """Security-with-abort driver: on abort, re-run the round with a fresh committee
    (seeded by the retry counter) that excludes the members suspected so far.

    Every retry is a fresh committee draw the adversary can trigger by
    misbehaving, so Lemma 2's bound is multiplied by (max_retries + 1)."""
    exclude: set = set(net.banned)
    nbytes: dict = defaultdict(int)
    tphase: dict = defaultdict(float)
    res = None
    captured = breach = False
    for attempt in range(net.cfg.max_retries + 1):
        res = net.run_round(rnd, updates, center, attempt=attempt, exclude=exclude, **kw)
        # shares were sent in every attempt: exposure accumulates over attempts
        captured |= res.captured
        breach |= res.privacy_breach
        for k, v in res.bytes_by_kind.items():
            nbytes[k] += v
        for k, v in res.time_by_phase.items():
            tphase[k] += v
        for sus in res.suspects:
            if len(net.banned) < net.ban_cap:
                net.banned.add(sus)
        if not res.aborted:
            break
        exclude |= res.suspects
    res.bytes_by_kind, res.time_by_phase, res.attempts = dict(nbytes), dict(tphase), attempt + 1
    res.captured, res.privacy_breach = captured, breach
    res.banned = len(net.banned)
    return res
