# Importing here registers every built-in tool with the shared registry
# (each tool module calls @registry.register(...) at import time).
from app.tools import code_edit, code_rag, code_search, filesystem, git_tools, testing  # noqa: F401,E402