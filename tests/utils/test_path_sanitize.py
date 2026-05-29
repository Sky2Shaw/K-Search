from k_search.utils.path_sanitize import sanitize_worktree_paths


def test_replaces_single_worktree_path():
    text = "[workdir] /tmp/ksearch_agentic_worktree_02ut4r9r/tile2asc/mqa/kernel/x.cpp"
    out = sanitize_worktree_paths(text)
    assert "ksearch_agentic_worktree_02ut4r9r" not in out
    assert out == "[workdir] <PROJECT_ROOT>/tile2asc/mqa/kernel/x.cpp"


def test_replaces_temp_repo_fallback_path():
    text = "see /tmp/ksearch_agentic_temp_repo_abc123/kernel/foo.h"
    out = sanitize_worktree_paths(text)
    assert "ksearch_agentic_temp_repo_abc123" not in out
    assert out == "see <PROJECT_ROOT>/kernel/foo.h"


def test_replaces_multiple_distinct_random_suffixes():
    text = (
        "old /tmp/ksearch_agentic_worktree_02ut4r9r/a.cpp "
        "new /tmp/ksearch_agentic_worktree_14595g7x/b.cpp"
    )
    out = sanitize_worktree_paths(text)
    assert "02ut4r9r" not in out
    assert "14595g7x" not in out
    assert out.count("<PROJECT_ROOT>") == 2


def test_bare_worktree_root_replaced():
    text = "/tmp/ksearch_agentic_worktree_gvc"
    out = sanitize_worktree_paths(text)
    assert out == "<PROJECT_ROOT>"


def test_no_match_returns_unchanged():
    text = "compile ok, no paths here"
    assert sanitize_worktree_paths(text) == text


def test_idempotent():
    text = "/tmp/ksearch_agentic_worktree_xyz/kernel/x.cpp"
    once = sanitize_worktree_paths(text)
    assert sanitize_worktree_paths(once) == once


def test_custom_placeholder():
    text = "/tmp/ksearch_agentic_worktree_xyz/x.cpp"
    out = sanitize_worktree_paths(text, placeholder="<ROOT>")
    assert out == "<ROOT>/x.cpp"
