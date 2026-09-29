from dotenv import load_dotenv

load_dotenv()

import app.tools
from app.runtime.agent import Agent

agent = Agent()

result = agent.run(
    task="Run the test suite and report the results.",
    repository=".",
)

print(result.final_result)