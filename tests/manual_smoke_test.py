from dotenv import load_dotenv

load_dotenv()

import app.tools
from app.runtime.agent import Agent

agent = Agent()

result = agent.run(
    task="Check git status, create a new branch called 'forgeai/test-run-2', then commit all current changes with an appropriate message.",
    repository=".",
)

print(result.final_result)