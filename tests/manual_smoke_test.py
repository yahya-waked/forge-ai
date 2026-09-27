from app.runtime.agent import Agent 
agent = Agent()
result = agent.run(
    task="Check git status, create a new branch called 'forgeai/test-run', then commit all current changes with an appropriate message.",
    repository=".",
)
print(result.final_result)