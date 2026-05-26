import os

from k_search.kernel_generators.project_snapshot import (
    create_project_snapshot,
    materialize_project_snapshot,
)


def test_project_snapshot_records_large_files_modes_and_symlinks(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "kernel").mkdir()
    (project / "kernel" / "foo.h").write_text("int foo = 1;\n", encoding="utf-8")
    large = project / "kernel" / "large_header.hpp"
    large.write_text("x" * 210_000, encoding="utf-8")
    script = project / "bench.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    (project / "build").mkdir()
    (project / "build" / "artifact.o").write_text("ignored\n", encoding="utf-8")
    os.symlink("kernel/foo.h", project / "foo_link.h")

    snapshot = create_project_snapshot(
        project_dir=project,
        snapshot_id="snap_1",
        parent_snapshot_id=None,
        base_commit="abc123",
        created_by_round=3,
        eval_result={"status": "passed"},
    )

    assert set(snapshot.manifest) == {
        "bench.sh",
        "foo_link.h",
        "kernel/foo.h",
        "kernel/large_header.hpp",
    }
    assert snapshot.manifest["kernel/large_header.hpp"].size == 210_000
    assert snapshot.manifest["foo_link.h"].kind == "symlink"
    assert snapshot.manifest["foo_link.h"].link_target == "kernel/foo.h"
    assert snapshot.manifest["bench.sh"].mode.endswith("755")

    materialized = tmp_path / "materialized"
    materialize_project_snapshot(snapshot, materialized)

    assert (materialized / "kernel" / "large_header.hpp").read_text(encoding="utf-8") == "x" * 210_000
    assert os.readlink(materialized / "foo_link.h") == "kernel/foo.h"
    assert os.access(materialized / "bench.sh", os.X_OK)
    assert not (materialized / "build").exists()
