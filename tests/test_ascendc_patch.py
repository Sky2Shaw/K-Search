import pytest

from k_search.tasks.ascendc_patch import (
    ASCENDC_PATCH_FORMAT_TEXT,
    apply_unified_diff,
    parse_ascendc_project_patch,
)


def test_apply_unified_diff_modifies_single_hunk():
    base = "alpha\nbeta\ngamma\ndelta\nepsilon\n"
    diff = (
        "@@ -1,5 +1,5 @@\n"
        " alpha\n"
        " beta\n"
        "-gamma\n"
        "+GAMMA\n"
        " delta\n"
        " epsilon\n"
    )
    assert apply_unified_diff(base, diff) == "alpha\nbeta\nGAMMA\ndelta\nepsilon\n"


def test_apply_unified_diff_supports_pure_insertion_and_pure_deletion():
    base = "line1\nline2\nline3\n"
    insert_diff = (
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        " line2\n"
        "+inserted\n"
        " line3\n"
    )
    assert apply_unified_diff(base, insert_diff) == "line1\nline2\ninserted\nline3\n"

    delete_diff = (
        "@@ -1,3 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        " line3\n"
    )
    assert apply_unified_diff(base, delete_diff) == "line1\nline3\n"


def test_apply_unified_diff_raises_on_context_mismatch():
    base = "alpha\nbeta\ngamma\n"
    diff = (
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-WRONG_CONTEXT\n"
        "+new\n"
        " gamma\n"
    )
    with pytest.raises(ValueError) as exc_info:
        apply_unified_diff(base, diff)
    assert "context mismatch" in str(exc_info.value).lower()


def test_apply_unified_diff_normalizes_crlf():
    base = "alpha\r\nbeta\r\ngamma\r\n"
    diff = (
        "@@ -1,3 +1,3 @@\r\n"
        " alpha\r\n"
        "-beta\r\n"
        "+BETA\r\n"
        " gamma\r\n"
    )
    assert apply_unified_diff(base, diff) == "alpha\nBETA\ngamma\n"


def test_parse_ascendc_project_patch_applies_hunks_against_baseline():
    base_files = {
        "kernel/foo.h": "int a = 1;\nint b = 2;\nint c = 3;\n",
        "kernel/bar.cpp": "void run() {}\n",
    }
    raw = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " int a = 1;\n"
        "-int b = 2;\n"
        "+int b = 22;\n"
        " int c = 3;\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    files = parse_ascendc_project_patch(raw, base_files=base_files)
    assert files["kernel/foo.h"] == "int a = 1;\nint b = 22;\nint c = 3;\n"
    assert files["kernel/bar.cpp"] == "void run() {}\n"


def test_parse_ascendc_project_patch_supports_op_replace_for_full_rewrite():
    base_files = {"kernel/foo.h": "old\n"}
    raw = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h" op="replace">\n'
        "completely\nnew\ncontent\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    files = parse_ascendc_project_patch(raw, base_files=base_files)
    assert files["kernel/foo.h"] == "completely\nnew\ncontent\n"


def test_parse_ascendc_project_patch_creates_new_file_when_baseline_missing():
    base_files = {"kernel/foo.h": "existing\n"}
    raw = (
        "<ascendc_patch>\n"
        '<patch path="kernel/new_file.h" op="replace">\n'
        "brand new\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    files = parse_ascendc_project_patch(raw, base_files=base_files)
    assert files["kernel/new_file.h"] == "brand new\n"
    assert files["kernel/foo.h"] == "existing\n"


def test_parse_ascendc_project_patch_raises_value_error_when_container_missing():
    with pytest.raises(ValueError):
        parse_ascendc_project_patch("garbage with no patch tags", base_files={})


def test_ascendc_patch_format_text_documents_unified_diff_and_replace_op():
    assert "<ascendc_patch>" in ASCENDC_PATCH_FORMAT_TEXT
    assert "@@" in ASCENDC_PATCH_FORMAT_TEXT
    assert 'op="replace"' in ASCENDC_PATCH_FORMAT_TEXT


def test_apply_unified_diff_supports_beginning_of_file_insertion():
    base = "line1\nline2\n"
    diff = (
        "@@ -0,0 +1,2 @@\n"
        "+inserted_at_top_1\n"
        "+inserted_at_top_2\n"
    )
    result = apply_unified_diff(base, diff)
    assert result == "inserted_at_top_1\ninserted_at_top_2\nline1\nline2\n"


def test_parse_ascendc_project_patch_error_message_includes_file_path():
    base_files = {"kernel/foo.h": "alpha\nbeta\ngamma\n"}
    raw = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-WRONG\n"
        "+BETA\n"
        " gamma\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    with pytest.raises(ValueError) as exc_info:
        parse_ascendc_project_patch(raw, base_files=base_files)
    assert "kernel/foo.h" in str(exc_info.value)


def test_parse_ascendc_project_patch_ignores_draft_patches_outside_final_container():
    base_files = {"kernel/foo.h": "alpha\nbeta\ngamma\n"}
    raw = (
        "The compile failure is clear. Here is a draft patch:\n"
        "```patch\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-WRONG_DRAFT_CONTEXT\n"
        "+draft\n"
        " gamma\n"
        "</patch>\n"
        "```\n"
        "Final patch:\n"
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-beta\n"
        "+BETA\n"
        " gamma\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )

    with pytest.warns(RuntimeWarning, match="non-strict ascendc patch output"):
        files = parse_ascendc_project_patch(raw, base_files=base_files)

    assert files["kernel/foo.h"] == "alpha\nBETA\ngamma\n"


def test_parse_ascendc_project_patch_uses_last_container_when_multiple_are_present():
    base_files = {"kernel/foo.h": "alpha\nbeta\ngamma\n"}
    raw = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-WRONG_DRAFT_CONTEXT\n"
        "+draft\n"
        " gamma\n"
        "</patch>\n"
        "</ascendc_patch>\n"
        "The final corrected patch is below.\n"
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-beta\n"
        "+BETA\n"
        " gamma\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )

    with pytest.warns(RuntimeWarning, match="non-strict ascendc patch output"):
        files = parse_ascendc_project_patch(raw, base_files=base_files)

    assert files["kernel/foo.h"] == "alpha\nBETA\ngamma\n"
