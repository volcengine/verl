import subprocess
import sys
from pathlib import Path


def test_checkpoint_view_updates_code_without_modifying_source(tmp_path):
    source, target, code = (tmp_path / name for name in ("source", "view", "code"))
    source.mkdir()
    code.mkdir()
    (source / "config.json").write_text('{"model_type": "kimi_k3"}')
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "modeling_kimi_k3.py").write_text("original")
    (code / "modeling_kimi_k3.py").write_text("corrected")
    script = Path(__file__).resolve().parents[2] / "examples/ascend_extras/grpo_trainer/prepare_kimi_k3_model.py"
    command = [sys.executable, str(script), str(source), str(target), "--code", str(code)]
    subprocess.run(command, check=True)
    subprocess.run(command, check=True)
    assert (target / "model.safetensors").resolve() == source / "model.safetensors"
    assert not (target / "config.json").is_symlink()
    assert (target / "config.json").read_bytes() == (source / "config.json").read_bytes()
    assert (target / "modeling_kimi_k3.py").read_text() == "corrected"
    assert (source / "modeling_kimi_k3.py").read_text() == "original"

    command[3] = str(source)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Model view must be separate" in result.stderr
