import pytest
from pydantic import ValidationError

from eegkit.config import Config, load_config


def test_defaults_valid():
    c = Config()
    assert c.subjects == list(range(1, 11))
    assert c.analysis.bands["mu"] == (8.0, 13.0)


def test_load_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("subjects: [1, 2]\npreprocess: {l_freq: 2.0}\n")
    c = load_config(p)
    assert c.subjects == [1, 2] and c.preprocess.l_freq == 2.0


def test_shipped_default_yaml_loads():
    assert load_config("configs/default.yaml").subjects == list(range(1, 11))


@pytest.mark.parametrize(
    "bad",
    [
        {"subjects": []},
        {"subjects": [1, 1]},
        {"subjects": [0]},
        {"runs": [3]},
        {"preprocess": {"l_freq": 40, "h_freq": 1}},
        {"epochs": {"tmin": 2, "tmax": 1}},
        {"epochs": {"baseline": [-5, 0]}},
        {"analysis": {"bands": {"mu": [13, 8]}}},
        {"unknown_key": 1},
    ],
)
def test_invalid_rejected(bad):
    with pytest.raises(ValidationError):
        Config.model_validate(bad)
