# Importing here registers every built-in tool with the shared registry
# (each tool module calls @registry.register(...) at import time).
from app.tools import (  # noqa: F401,E402
    code_edit,
    code_rag,
    code_search,
    filesystem,
    git_tools,
    security_scan,
    testing,
)