"""Every aggregator runs end to end through the simulator (one round, tiny data)."""
import pytest

from pqbrdfl.fl import data, sim

DS = data.synthetic_sdn(n=3000, n_features=12)
AGGS = ["central", "fedavg", "dfl_gossip", "median", "trimmed_mean", "krum", "fltrust", "cc",
        "mk_cc", "mk_cc_exact", "cc_sf", "pqbrdfl", "pqbrdfl_cc", "pqbrdfl_mean", "pqbrdfl_norange",
        "pqbrdfl_cc_norange"]


@pytest.mark.parametrize("agg", AGGS)
def test_aggregator_runs(agg):
    r = sim.run(sim.SimConfig(aggregator=agg, attack="scaled_backdoor", n_agents=10, byz_frac=0.2,
                              rounds=1, eval_every=1, m=9, t=4), DS)
    assert 0.0 <= r["final"]["accuracy"] <= 1.0
