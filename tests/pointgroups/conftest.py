from dataclasses import dataclass
from pathlib import Path
import shutil

import pytest


@dataclass(frozen=True)
class PointGroupTestPaths:
    data_dir: Path
    cache_dir: Path

    def cached_path(self, *parts: str) -> Path:
        path = self.cache_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def cached_dir(self, *parts: str) -> Path:
        path = self.cache_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def tmole_json_path(self, name: str) -> Path:
        checked_in = self.data_dir / f"{name}_tmole.json"
        if checked_in.exists():
            return checked_in
        return self.cached_path("tmole-json", f"{name}_tmole.json")

    def migrate_legacy_cache(self) -> None:
        file_moves = {
            self.data_dir / "Al13_wfs.gpw": self.cached_path("test_molecules", "Al13_wfs.gpw"),
            self.data_dir / "gs_Oh.gpw": self.cached_path("test_pointgroup", "gs_Oh.gpw"),
            self.data_dir / "gpaw.txt": self.cached_path("test_molecules", "gpaw.txt"),
        }

        for source, target in file_moves.items():
            if source.exists() and not target.exists():
                source.replace(target)

        for source in self.data_dir.glob("MoS2_test_*.gpw"):
            target = self.cached_path("test_pointgroup", source.name)
            if not target.exists():
                source.replace(target)

        for source in self.data_dir.glob("cache_*.gpw"):
            target = self.cached_path("test_pointgroup", source.name)
            if not target.exists():
                source.replace(target)

        molecule_cache = self.cached_dir("molecules")
        for source in self.data_dir.iterdir():
            if not source.is_dir():
                continue
            if source.name in {"Oh", "structures", "__pycache__"}:
                continue
            target = molecule_cache / source.name
            if not target.exists():
                shutil.move(str(source), str(target))


@pytest.fixture(scope="session")
def pointgroup_test_paths(pytestconfig) -> PointGroupTestPaths:
    data_dir = Path(__file__).resolve().parent
    cache_dir = Path(pytestconfig.cache.mkdir("pointgroups"))
    paths = PointGroupTestPaths(data_dir=data_dir, cache_dir=cache_dir)
    paths.migrate_legacy_cache()
    return paths


@pytest.fixture(scope="session")
def pointgroup_cache_dir(pointgroup_test_paths: PointGroupTestPaths) -> Path:
    return pointgroup_test_paths.cache_dir


@pytest.fixture(autouse=True)
def work_from_test_data_dir(monkeypatch, pointgroup_test_paths: PointGroupTestPaths):
    monkeypatch.chdir(pointgroup_test_paths.data_dir)
