import os
from crewai import Agent, Task, Crew, Process, LLM

os.environ["OPENAI_API_KEY"] = "NA"

def run_consensus(context_data, user_query):
    """Takes retrieved PDF text and runs it through the Analyst and Auditor."""
    
    llama = LLM(
        model="ollama/llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0.1
    )

    extractor = Agent(
        role='Lead Data Analyst',
        goal='Extract facts from the provided context to answer the user query.',
        backstory='You are an expert data analyst. You only use the provided context. If the answer is not in the context, you state that clearly.',
        verbose=True,
        llm=llama
    )

    auditor = Agent(
        role='Quality Assurance Auditor',
        goal='Cross-examine the analyst\'s summary against the original context.',
        backstory='You are a rigorous QA engineer. You find missing details, hallucinations, or logic gaps. You output the FINAL verified answer.',
        verbose=True,
        llm=llama
    )

    task_extract = Task(
        description=f'Context Data:\n{context_data}\n\nUser Query: {user_query}\n\nDraft a comprehensive summary answering the query using ONLY the context.',
        expected_output='A drafted summary answering the query.',
        agent=extractor
    )

    task_audit = Task(
        description=f'Review the analyst\'s summary against this original Context:\n{context_data}\n\nRefine the answer, correct any mistakes, and output the final, polished response for the user.',
        expected_output='The final, verified, and well-formatted answer to the user query.',
        agent=auditor
    )

    crew = Crew(
        agents=[extractor, auditor],
        tasks=[task_extract, task_audit],
        process=Process.sequential
    )

    crew.kickoff()
    
    # NEW: Return a dictionary capturing the internal debate stages
    return {
        "analyst_draft": str(task_extract.output),
        "final_answer": str(task_audit.output)
    }