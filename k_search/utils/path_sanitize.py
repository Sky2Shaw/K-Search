from __future__ import annotations

import re

# 匹配 ksearch 临时 worktree / fallback 临时 repo 的绝对路径根:
#   /<任意前缀>/ksearch_agentic_worktree_<rand>
#   /<任意前缀>/ksearch_agentic_temp_repo_<rand>
# 捕获到随机根目录为止,其后的相对子路径保留不动。
# 后缀 [A-Za-z0-9]+ 对应 tempfile.mkdtemp 产出的随机名(见 k_search/kernel_generators/agentic_worktree.py 的
# prefix="ksearch_agentic_worktree_" / "ksearch_agentic_temp_repo_",无 suffix= 故为纯字母数字)。
_WORKTREE_ROOT_RE = re.compile(
    r"/[^\s]*?/(?:ksearch_agentic_worktree|ksearch_agentic_temp_repo)_[A-Za-z0-9]+"
)


def sanitize_worktree_paths(text: str, *, placeholder: str = "<PROJECT_ROOT>") -> str:
    """把任意 ksearch 临时 worktree / 临时 repo 的绝对路径前缀替换为语义占位符。

    用通配正则而非精确字符串,故任意历史轮次的残留路径都会被替换,无需知道
    "当前 worktree 是谁";天然幂等。
    """
    if not text:
        return text
    return _WORKTREE_ROOT_RE.sub(placeholder, text)
