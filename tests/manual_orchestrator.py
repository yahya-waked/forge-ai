"""Manual end-to-end run of the multi-agent pipeline (calls the real LLM)."""
from dotenv import load_dotenv

load_dotenv()

from app.agents.orchestrator import Orchestrator  # noqa: E402

orchestrator = Orchestrator()
result = orchestrator.run(
           task=(
        "Add a subtract(a, b) function to buggy_math.py that returns a - b, "
        "with a matching test in test_buggy_math.py. Make sure all tests pass."
        ),
    repository=".",
)

print("\n=== STATUS:", result.status, "| review rounds:", result.rounds)
print("\n--- Review ---\n", result.review)
print("\n--- Security ---\n", result.security)
print("\n--- Summary ---\n", result.summary)