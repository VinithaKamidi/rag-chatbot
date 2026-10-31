import os

import config


def test_dotenv_loads_values_but_never_overrides_real_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nA_NEW_KEY='quoted value'\nEXISTING=from_file\nEMPTY=\nnot a pair\n")
    monkeypatch.delenv("A_NEW_KEY", raising=False)
    monkeypatch.setenv("EXISTING", "from_shell")
    monkeypatch.delenv("EMPTY", raising=False)

    config.load_dotenv(env)

    assert os.environ["A_NEW_KEY"] == "quoted value"
    assert os.environ["EXISTING"] == "from_shell"
    assert "EMPTY" not in os.environ
    monkeypatch.delenv("A_NEW_KEY")


def test_missing_dotenv_is_fine(tmp_path):
    config.load_dotenv(tmp_path / "nope.env")  # must not raise
