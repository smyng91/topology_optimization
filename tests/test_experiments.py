from pathlib import Path

import pytest

from topoopt.experiments import (
    PROTOCOL_PATH,
    case_fingerprint,
    case_params,
    fingerprints_match,
    load_protocol,
)

if not PROTOCOL_PATH.is_file():
    pytest.skip("experiments protocol is local-only", allow_module_level=True)


def test_protocol_loads_registered_cases():
    protocol = load_protocol()
    assert protocol["protocol_id"] == "journal-neutral-2d-v1"
    params = case_params(protocol, "tree")
    assert params.n == (80, 80)
    assert params.heat_mode == "conduction"


def test_fingerprint_changes_with_source_digest():
    protocol = load_protocol()
    left = case_fingerprint(protocol, "darcy", seed=0, model_source_sha256="aaa")
    right = case_fingerprint(protocol, "darcy", seed=0, model_source_sha256="bbb")
    assert fingerprints_match(left, left)
    assert not fingerprints_match(left, right)
    other_seed = case_fingerprint(protocol, "darcy", seed=1, model_source_sha256="aaa")
    assert not fingerprints_match(left, other_seed)
