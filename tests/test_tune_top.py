"""Tuning resume, tuned-model loading, and the AUC bootstrap behind the comparison."""
import numpy as np
import pytest
import torch

from experiments.score_risk import load_notebook_model
from experiments.tune_top import requeue_interrupted, uses_dropout
from src.config import DEVICE, K, STATE_DIM
from src.evaluation import collision as C
from src.models.strap import STRAP
from src.train import fit
from tests.test_train import _sample

STATS = {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)}


@pytest.mark.parametrize("higher", [True, False])
def test_placements_average_to_the_roc_auc_with_ties_and_infinities(higher):
    rng = np.random.default_rng(0)
    scores = rng.normal(size=4000).round(1)               # heavy ties
    labels = (rng.random(4000) < 0.05 + 0.2 * (scores > 1)).astype(int)
    scores[:40] = np.inf
    placements = C.auc_placements(scores, labels, higher)
    assert placements.shape == (labels.sum(),)
    assert np.isclose(placements.mean(), C.compute_roc_auc(scores, labels, higher))


def test_event_bootstrap_brackets_the_estimate_and_pairs_draws():
    rng = np.random.default_rng(1)
    placements = rng.random(300)
    events = rng.integers(0, 40, 300)
    (lo, hi), = C.event_bootstrap(placements, events)
    assert lo < placements.mean() < hi
    # one event repeated: nothing to resample, so the interval has no width
    (lo, hi), = C.event_bootstrap(placements, np.zeros(300))
    assert np.isclose(lo, hi)
    # the same draw is applied to every row, so a model against itself is identical
    both = C.event_bootstrap(np.stack([placements, placements]), events)
    assert np.allclose(both[0], both[1])


@pytest.mark.parametrize("name", ["inatran", "cognitive", "miat"])
def test_dropout_override_reaches_the_model(name):
    assert uses_dropout(name)
    model, _ = load_notebook_model(name, device=torch.device("cpu"), load_weights=False,
                                   stats=STATS, overrides={"DROPOUT": 0.27})
    rates = {m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)}
    assert rates == {0.27}


def test_models_without_dropout_do_not_tune_it():
    assert not any(uses_dropout(n) for n in ("dacr_amtp", "stdan", "mha_lstm",
                                             "x_track", "traj_mamba"))


def test_an_interrupted_trial_is_requeued_once_with_its_params(tmp_path):
    """Kill -> RUNNING; Ctrl-C -> FAIL with the checkpoint left. Either way the
    newest trial of the chain is re-queued once, pointing at the first trial's dir."""
    import optuna
    from optuna.trial import TrialState

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(storage=f"sqlite:///{(tmp_path / 'db').as_posix()}")
    first = study.ask({"lr": optuna.distributions.FloatDistribution(1e-4, 1e-2)})
    assert requeue_interrupted(study, tmp_path) == 1        # killed while RUNNING
    assert requeue_interrupted(study, tmp_path) == 0        # already WAITING

    resumed = study.ask()                                   # the re-queued trial
    assert resumed.params == first.params
    assert resumed.user_attrs["resume_of"] == first.number
    (tmp_path / f"trial_{first.number}").mkdir()
    (tmp_path / f"trial_{first.number}" / "last_checkpoint.pt").touch()
    study.tell(resumed, state=TrialState.FAIL)              # Ctrl-C, checkpoint kept
    assert requeue_interrupted(study, tmp_path) == 1
    assert requeue_interrupted(study, tmp_path) == 0
    waiting = [t for t in study.trials if t.state == TrialState.WAITING]
    assert [t.user_attrs["resume_of"] for t in waiting] == [first.number]


def test_fit_resumes_from_the_last_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    centers = np.random.default_rng(0).normal(size=(K, 4)).astype(np.float32)
    train, val = [_sample() for _ in range(4)], [_sample() for _ in range(2)]
    common = dict(learning_rate=1e-3, batch_size=2, save_dir=str(tmp_path / "ckpt"))

    fit(STRAP().to(DEVICE), train, val, centers, num_epochs=1, **common)
    log = fit(STRAP().to(DEVICE), train, val, centers, num_epochs=2,
              resume_from=str(tmp_path / "ckpt" / "last_checkpoint.pt"), **common)
    assert [entry["epoch"] for entry in log] == [1, 2]
