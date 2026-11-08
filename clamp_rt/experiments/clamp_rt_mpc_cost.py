"""Cost of the CLAMP-RT server step in two-party MPC (CrypTen, semi-honest,
arithmetic sharing over Z_{2^64}, 16 fractional bits, Beaver triples from a
trusted dealer).

Two processes on one machine play servers A and B. Per round they

  1. check ||u_k||^2 <= C^2 for every client (one secure dot product and one
     comparison per client),
  2. check -1 <= d_k <= 1 (two comparisons per client),
  3. form the masked update sum sum_k v_k u_k, with v_k the shared validity bit,
     and open it,
  4. replace rejected gaps by 0, sort the K gaps with an odd-even transposition
     network, average the K - 2b middle values, add one Gaussian share per server
     and open the result.

Client-to-server input sharing is not timed (it is 2 x 8 bytes per value per
client, reported analytically). Reported per configuration: wall time of
steps 1-4, bytes sent per server and communication rounds, separately for the
update channel (1+3) and the gap channel (2+4). WAN time is estimated as
local time + rounds * RTT + bytes / bandwidth.

Usage:  python experiments/clamp_rt_mpc_cost.py --out results/clamp_rt/mpc_cost.json
(run in an environment with CrypTen installed).
"""
import argparse
import json
import math
import os
import time

import torch


def _stats():
    import crypten.communicator as comm
    s = comm.get().get_communication_stats()
    return s["bytes"], s["rounds"]


def _reset():
    import crypten.communicator as comm
    comm.get().reset_communication_stats()


def _sort_network(x):
    """Odd-even transposition sort of a 1-D shared vector (K passes)."""
    import crypten
    K = x.size(0)
    vals = [x[i] for i in range(K)]
    for p in range(K):
        start = p % 2
        pairs = [(i, i + 1) for i in range(start, K - 1, 2)]
        if not pairs:
            continue
        a = crypten.stack([vals[i] for i, _ in pairs])
        b = crypten.stack([vals[j] for _, j in pairs])
        c = a > b                       # shared bit, one comparison per pair
        lo = a + c * (b - a)
        hi = a + b - lo
        for n, (i, j) in enumerate(pairs):
            vals[i], vals[j] = lo[n], hi[n]
    return crypten.stack(vals)


def _round_step(K, d, b, C, sigma, seed):
    import crypten
    torch.manual_seed(seed)
    # Plaintext inputs (played by "clients"); honest-looking data plus one oversized update.
    U = torch.randn(K, d)
    U = U / U.norm(dim=1, keepdim=True) * (0.2 + 0.7 * torch.rand(K, 1)) * C
    U[0] *= 1.5 / float(U[0].norm())                  # one rejected update (norm 1.5 C)
    # Clients share s * u with s = sqrt(d) / C, so entries are O(1) in fixed point
    # and the check becomes ||s u||^2 <= d.
    s = math.sqrt(d) / C
    g = (torch.rand(K) - 0.5) * 0.2
    g[1] = 1.7                                        # one out-of-range gap
    Us = crypten.cryptensor(U * s, src=0)
    gs = crypten.cryptensor(g, src=0)

    # Gap range check (gap channel).
    _reset(); t0 = time.perf_counter()
    v_g = (gs >= -1.0) * (gs <= 1.0)
    t_g1 = time.perf_counter() - t0
    by_g1, r_g1 = _stats()

    # Update channel: norm check and masked sum over clients valid on both checks.
    _reset(); t0 = time.perf_counter()
    sq = (Us * Us).sum(dim=1)                         # K secure squared norms
    v_u = sq <= float(d)
    valid = v_u * v_g
    agg = (Us * valid.unsqueeze(1)).sum(dim=0)
    agg_plain = agg.get_plain_text() / s
    t_upd = time.perf_counter() - t0
    by_u, r_u = _stats()

    # Gap channel: zero rejected gaps, sort, trim, noise.
    _reset(); t0 = time.perf_counter()
    gz = gs * valid
    srt = _sort_network(gz)
    tm = srt[b:K - b].mean()
    noise = crypten.cryptensor(torch.randn(1) * sigma, src=0) + \
        crypten.cryptensor(torch.randn(1) * sigma, src=1)
    out = (tm + noise[0]).get_plain_text()
    t_gap = time.perf_counter() - t0 + t_g1
    by_g, r_g = _stats()
    by_g, r_g = by_g + by_g1, r_g + r_g1

    # Plaintext reference for the correctness check.
    ok = ((U.norm(dim=1) <= C) & (g.abs() <= 1)).float()
    ref_tm = torch.sort(g * ok).values[b:K - b].mean()
    ref_sum = (U * ok.unsqueeze(1)).sum(0)
    err_sum = float((agg_plain - ref_sum).abs().max())
    return dict(K=K, d=d, b=b, t_update_s=t_upd, bytes_update=by_u, rounds_update=r_u,
                t_gap_s=t_gap, bytes_gap=by_g, rounds_gap=r_g,
                max_abs_err_sum=err_sum, tm_ref=float(ref_tm),
                tm_noisy=float(out), sigma=sigma)


def _worker(configs, out_path, reps):
    import crypten
    crypten.cfg.communicator.verbose = True
    import crypten.communicator as comm
    rank = comm.get().get_rank()
    rows = []
    for (K, d) in configs:
        b = max(1, int(math.floor(0.2 * K)))
        res = [_round_step(K, d, b, 1.0, 0.0, seed=r) for r in range(reps)]
        row = {k: res[0][k] for k in ("K", "d", "b", "bytes_update", "rounds_update",
                                      "bytes_gap", "rounds_gap")}
        for k in ("t_update_s", "t_gap_s"):
            row[k] = sorted(r[k] for r in res)[len(res) // 2]     # median over reps
        row["max_abs_err_sum"] = max(r["max_abs_err_sum"] for r in res)
        row["max_abs_err_tm"] = max(abs(r["tm_noisy"] - r["tm_ref"]) for r in res)
        rows.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)
    if rank == 0:
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/clamp_rt/mpc_cost.json")
    ap.add_argument("--K", default="5,10,20")
    ap.add_argument("--d", default="10000,100000,1000000")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    configs = [(int(K), int(d)) for K in args.K.split(",") for d in args.d.split(",")]

    import crypten
    from crypten import mpc
    crypten.init()

    @mpc.run_multiprocess(world_size=2)
    def run():
        _worker(configs, args.out, args.reps)

    run()


if __name__ == "__main__":
    main()
