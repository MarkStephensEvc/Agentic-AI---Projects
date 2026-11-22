from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent# TODO: 1 - Import EvaluationAgent and KnowledgeAugmentedPromptAgent classes
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
prompt = "What is the capital of France?"

# Parameters for the Knowledge Agent
persona1 = "a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key,persona1,knowledge) # TODO: 2 - Instantiate the KnowledgeAugmentedPromptAgent here
# Parameters for the Evaluation Agent
persona2 = "an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence, and provide factually accurate information"
evaluation_agent = EvaluationAgent(openai_api_key,persona2,evaluation_criteria,knowledge_agent,max_interactions=10) # TODO: 3 - Instantiate the EvaluationAgent with a maximum of 10 interactions here

response = evaluation_agent.evaluate(prompt)# TODO: 4 - Evaluate the prompt and print the response from the EvaluationAgent
print(response)