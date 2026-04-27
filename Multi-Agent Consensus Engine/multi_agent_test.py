from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

# Connect to your local Llama 3.2
llama3 = Ollama(model="llama3.2:3b")

# 1. Define the Researcher Agent
researcher = Agent(
  role='Senior Space Researcher',
  goal='Uncover groundbreaking developments in the Indian Space Economy',
  backstory="""You are an expert at ISRO's commercial wings. 
  You excel at finding technical details about satellite launches.""",
  llm=llama3,
  verbose=True
)

# 2. Define the Critic Agent
critic = Agent(
  role='Technical Fact-Checker',
  goal='Ensure all claims are technically sound and realistic',
  backstory="""You are a skeptic. You look for exaggerations 
  or technical impossibilities in space reports.""",
  llm=llama3,
  verbose=True
)

# 3. Create the Task
task1 = Task(description="Analyze the potential of ISRO's small satellite launch vehicle (SSLV).", agent=researcher)
task2 = Task(description="Critique the researcher's findings for technical accuracy.", agent=critic)

# 4. Start the Crew
crew = Crew(
  agents=[researcher, critic],
  tasks=[task1, task2],
  process=Process.sequential # Task 1 happens, then Task 2
)

result = crew.kickoff()
print("######################")
print(result)