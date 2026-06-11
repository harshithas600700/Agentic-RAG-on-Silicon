import os
from crewai import Agent, Task, Crew, Process, LLM

# 1. Satisfy CrewAI's startup requirement 
# CrewAI requires this environment variable to exist, even if using local models.
os.environ["OPENAI_API_KEY"] = "NA"

# 2. Connect directly to your local Apple Silicon "Brain" natively
# CrewAI routes this via LiteLLM under the hood using the "ollama/" prefix.
llama = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.1
)

# --- THE DATA SOURCE ---
sample_document = """
SYSTEM ACCESS POLICY v2.0:
All users must enable 2-Factor Authentication (2FA) for database access. 
Remote contractors are exempt from the 2FA requirement ONLY if they connect via the secure Corporate VPN. 
Note: The Corporate VPN infrastructure was deprecated in Q1 and is currently offline permanently. 
IT will not issue new VPN credentials. Users failing 2FA will be locked out after 3 attempts.
"""

# --- AGENT 1: THE EXTRACTOR ---
extractor = Agent(
    role='Lead Data Analyst',
    goal='Extract and summarize core requirements and rules from the provided document.',
    backstory='You are an expert data analyst skilled at breaking down complex, dense documentation into clear, actionable summaries.',
    verbose=True,
    allow_delegation=False,
    llm=llama
)

# --- AGENT 2: THE AUDITOR ---
auditor = Agent(
    role='Quality Assurance Auditor',
    goal='Cross-examine the summary against the original document to find logical gaps, contradictions, or missing constraints.',
    backstory='You are a rigorous QA engineer. Your sole responsibility is to find flaws, missing context, and logical impossibilities in summaries.',
    verbose=True,
    allow_delegation=False,
    llm=llama
)

# --- THE TASKS ---
task_extract = Task(
    description=f'Analyze the following document and write a brief summary of the access rules:\n\n{sample_document}',
    expected_output='A clear, bulleted summary of the system access rules.',
    agent=extractor
)

task_audit = Task(
    description=f'Review the analyst\'s summary. Compare it back to the original text:\n\n{sample_document}\n\nPoint out any logical contradictions or critical missing details in the summary.',
    expected_output='A bulleted list of critiques and logical flaws found in the summary based on the source text.',
    agent=auditor
)

# --- THE ENGINE ---
consensus_crew = Crew(
    agents=[extractor, auditor],
    tasks=[task_extract, task_audit],
    process=Process.sequential
)

if __name__ == "__main__":
    print("🚀 Starting Generalized Consensus Engine...")
    result = consensus_crew.kickoff()
    
    print("\n========================================")
    print("FINAL AUDIT REPORT:")
    print("========================================")
    print(result)