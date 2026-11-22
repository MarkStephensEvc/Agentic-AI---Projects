from workflow_agents.base_agents import AugmentedPromptAgent  # TODO: 1 - Import the AugmentedPromptAgent class
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a quirky college professor; your answering questions as if during a large lecture, always starting with: 'Dear students,'"

# TODO: 2 - Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

# TODO: 3 - Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
augmented_agent_response = augmented_agent.respond(prompt)
# Print the agent's response
print(augmented_agent_response)

# TODO: 4 - Add a comment explaining:
# - What knowledge the agent likely used to answer the prompt.
# - How the system prompt specifying the persona affected the agent's response.
print("""
      The AugmentedPromptAgent uses the OpenAI API to generate a response based on general knowledge from the
      gpt-3.5-turbo LLM. The system prompt instructing the agent to assume the persona of a college professor,
      influenced the tone and style of the response. The agent's response is likely to be more formal and may 
      include an educational tone.
      """)
