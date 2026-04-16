import subprocess
import os
from crewai import Agent, Task, Crew

# 1. RUN THE COMMAND MANUALLY IN PYTHON (REAL DATA)
try:
    cmd = "kubectl get ns"
    env = {**os.environ, "KUBECONFIG": "/home/crysis/.kube/config"}
    raw_output = subprocess.check_output(cmd, shell=True, text=True, env=env)
except Exception as e:
    raw_output = f"Error: {str(e)}"

# 2. GIVE THE REAL DATA TO THE AGENT
analyst = Agent(
    role='Infrastructure Analyst',
    goal='Format the provided raw Kubernetes data into a professional report.',
    backstory='You are an expert at taking raw terminal output and making it readable.',
    llm="ollama/qwen2.5-coder:7b",
    verbose=True
)

report_task = Task(
    description=(
        f"Here is the raw output from the cluster:\n\n{raw_output}\n\n"
        "Create a Markdown table showing the Pod Name and Status. "
        "Highlight any pods that are NOT 'Running'."
    ),
    expected_output="A clean Markdown table of the actual pods.",
    agent=analyst
)

crew = Crew(agents=[analyst], tasks=[report_task])
print(crew.kickoff())