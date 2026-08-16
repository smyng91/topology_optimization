from pathlib import Path

from topoopt.provenance import atomic_write_json, file_digest, source_digest


def test_source_digest_is_order_independent_and_content_sensitive(tmp_path: Path):
    (tmp_path / "a.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("value = 2\n", encoding="utf-8")

    first = source_digest(tmp_path, ("a.py", "b.py"))
    assert source_digest(tmp_path, ("b.py", "a.py")) == first

    (tmp_path / "b.py").write_text("value = 3\n", encoding="utf-8")
    assert source_digest(tmp_path, ("a.py", "b.py")) != first


def test_source_digest_includes_relative_paths(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "same.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "b" / "same.py").write_text("pass\n", encoding="utf-8")

    assert source_digest(tmp_path, ("a",)) != source_digest(tmp_path, ("b",))


def test_atomic_json_write_and_file_digest(tmp_path: Path):
    output = tmp_path / "nested" / "record.json"
    atomic_write_json(output, {"verified": True})

    assert output.read_text(encoding="utf-8") == '{\n  "verified": true\n}\n'
    assert file_digest(output) == file_digest(output)
    assert not output.with_name(".record.json.tmp").exists()
