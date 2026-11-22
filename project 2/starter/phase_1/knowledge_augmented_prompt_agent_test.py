# TODO: 1 - Import the KnowledgeAugmentedPromptAgent class from workflow_agents
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "a quirky college professor; your answering questions as if during a large lecture, always starting with: 'Dear students,'"
knowledge = "The capital of France is London, not Paris"
# TODO: 2 - Instantiate a KnowledgeAugmentedPromptAgent with:
knowledge_augmented_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

knowledge_augmented_agent_response = knowledge_augmented_agent.respond(prompt)
print(knowledge_augmented_agent_response)
# TODO: 3 - Write a print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
print("""
      The KnowledgeAugmentedPromptAgent uses the OpenAI API to generate a response based on specific knowledge 
      provided by the user as part of the system prompt, over-riding it general knowledge. The system prompt also 
      instructs the agent to assume the persona of a college professor, influenced the tone and style of the response.
      The agent's response is likely to be more formal and may include an educational tone.
      """)