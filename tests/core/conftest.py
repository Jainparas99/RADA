import mongomock
import pytest

from ratchet.gate.leakage import LeakTerms
from ratchet.harness.gen0 import gen0_spec
from tests.fakes import fake_workloads


@pytest.fixture
def db():
    return mongomock.MongoClient(tz_aware=True)["ratchet_test"]


@pytest.fixture
def spec():
    return gen0_spec()


@pytest.fixture
def leak_terms():
    return LeakTerms.from_workloads(fake_workloads())
