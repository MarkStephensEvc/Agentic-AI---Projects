from openai import OpenAI
import numpy as np
import pandas as pd
import re
import csv
import uuid
from enum import Enum
from datetime import datetime

class OpenAIModels(str, Enum):
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_41_MINI = "gpt-4.1-mini"
    GPT_41_NANO = "gpt-4.1-nano"
    
MODEL = OpenAIModels.GPT_35_TURBO

# DirectPromptAgent class definition
class DirectPromptAgent:
    
    def __init__(self, openai_api_key):
        # Initialize the agent
        # TODO: 2 - Define an attribute named openai_api_key to store the OpenAI API key provided to this class.
        self.openai_api_key = openai_api_key
        self.test = "This is a test"

    def respond(self, prompt):
        # Generate a response using the OpenAI API
        client = OpenAI(api_key=self.openai_api_key,
                        base_url="https://openai.vocareum.com/v1")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                # TODO: 4 - Provide the user's prompt here. Do not add a system prompt.
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        # TODO: 5 - Return only the textual content of the response (not the full JSON response).
        return response.choices[0].message.content

        
# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        # TODO: 1 - Create an attribute for the agent's persona
        self.openai_api_key = openai_api_key
        self.persona = persona

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        client = OpenAI(api_key=self.openai_api_key,
                        base_url="https://openai.vocareum.com/v1")

        # TODO: 2 - Declare a variable 'response' that calls OpenAI's API for a chat completion.
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                # TODO: 3 - Add a system prompt instructing the agent to assume the defined persona and explicitly forget previous context.
                {"role":"system","content": f"{self.persona}. Forget all previous context."},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )

        return response.choices[0].message.content


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        # TODO: 1 - Create an attribute to store the agent's knowledge.
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = OpenAI(api_key=self.openai_api_key,
                        base_url="https://openai.vocareum.com/v1")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                # TODO: 2 - Construct a system message including:
                #           - The persona with the following instruction:
                #             "You are _persona_ knowledge-based assistant. Forget all previous context."
                #           - The provided knowledge with this instruction:
                #             "Use only the following knowledge to answer, do not use your own knowledge: _knowledge_"
                #           - Final instruction:
                #             "Answer the prompt based on this knowledge, not your own."
                {"role": "system", "content": f"""
                 {self.persona} knowledge-based assistant. Forget all previous context.
                 Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}
                 Answer the prompt based on this knowledge, not your own.                 
                 """},               
                # TODO: 3 - Add the user's input prompt here as a user message.
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        print(f'fetching the embedding vector for chunk of text starting with {text[:10]}')
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        print('Calculating scosine simularity')
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        #cosine similarity
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        print(f'Cosine simularity metric is: {cos_theta: 0.2f}, closer to 1 is better')
        return cos_theta

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        print('chunking text.....')
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()
        #default
        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        #guard: overlap must be less than chunk_size
        safe_overlap = min(self.chunk_overlap, self.chunk_size - 1)
        #initialise variables
        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text) and separator in text[start:end]:
                candidate_end = start + text[start:end].rindex(separator) + len(separator)
                # Only use separator-based end if seperator isn't beginning the text
                if candidate_end > start + safe_overlap:
                    end = candidate_end
                
            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })
            

            next_start = end - safe_overlap
            # guard: ensures a hard advance
            if next_start <= start:
                next_start = end  

            start = next_start
            chunk_id += 1
            print(f"Chunk {chunk_id} completed! Progress: {min(start, len(text))*100/len(text):.2f}%")
        # write embeddings to file
        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})
                print(f"Text chunk ID {chunk['chunk_id']}: '{chunk['text'][:20]}....' of size {chunk['chunk_size']} written to csv")
        
        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        print('Chunk embeddings calculated.')
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']
        print('Best chunk obtained from highest similarity')

        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content

class EvaluationAgent:
    
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        # Initialize the EvaluationAgent with given attributes.
        # TODO: 1 - Declare class attributes here
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions
        
    def evaluate(self, initial_prompt):
        import json
        # This method manages interactions between agents to achieve a solution.
        client = OpenAI(api_key=self.openai_api_key,
                        base_url="https://openai.vocareum.com/v1")
        prompt_to_evaluate = initial_prompt
        response_from_worker = None
        #exceed max iteration output
        evaluation = {'verdict': 'No', 'reason': 'Max iterations reached without valid evaluation'}
        for i in range(self.max_interactions,): # TODO: 2 - Set loop to iterate up to the maximum number of interactions:
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)# TODO: 3 - Obtain a response from the worker agent
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria} "  # TODO: 4 - Insert evaluation criteria here
            )
            output_format = """Respond in JSON format:
            {"verdict": "Yes or No", "reason": "reason why it does or does not meet the criteria"}"""               
            
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                {"role": "system", "content": f"{self.persona}, a response evaluation assistant. Forget previous context."},
                {"role": "user", "content": f"{eval_prompt}.\n\n{output_format}"}
            ],
            temperature=0  # TODO: 5 - Define the message structure sent to the LLM for evaluation (use temperature=0)
            )
            try:
                evaluation = json.loads(response.choices[0].message.content.strip())
            except json.JSONDecodeError:
                print("Warning: Evaluator returned malformed JSON, retrying...")
                continue
            print(f"Evaluator Agent Evaluation:\n{evaluation}")
           
            print(" Step 3: Check if evaluation is positive")
            if evaluation['verdict'].lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation['reason']}"
                )
                response = client.chat.completions.create(
                    model= MODEL,
                    messages=[
                    {"role": "system", "content": f"{self.persona}, a response evaluation assistant. Forget previous context."},
                    {"role": "user", "content": f"{instruction_prompt}\n\nCONTEXT:\nThe evaluation provided was for the following:\n{eval_prompt}"}
                ],# TODO: 6 - Define the message structure sent to the LLM to generate correction instructions (use temperature=0)
                temperature = 0
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        return {
            'iteration':i + 1,
            'final_response': response_from_worker,
            'evaluation': f"Correct?: {evaluation['verdict']}. {evaluation['reason']}"
        }   

class RoutingAgent():

    def __init__(self, openai_api_key, agents):
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        self.agents = agents
        # TODO: 1 - Define an attribute to hold the agents, call it agents

    def get_embedding(self, text):
        client = OpenAI(api_key=self.openai_api_key,base_url="https://openai.vocareum.com/v1")
        # TODO: 2 - Write code to calculate the embedding of the text using the text-embedding-3-large model
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        # Extract and return the embedding vector from the response
        embedding = response.data[0].embedding
        return embedding 

    # TODO: 3 - Define a method to route user prompts to the appropriate agent
    def get_agent(self,user_prompt):
        # TODO: 4 - Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_prompt)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # TODO: 5 - Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent['description'])
            if agent_emb is None:
                continue
            #cosine similarity
            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(f'similarity score for {agent['name']} is {similarity}')
            #assign similarity score to agent dictionary
            agent['similarity'] = similarity
        #determine best agent based on greatest similarity score    
        best_agent = max(self.agents, key=lambda agent: agent['similarity'])
        print(f'Best agent obtained from highest similarity is {best_agent['name']}')# TODO: 6 - Add logic to select the best agent based on the similarity score between the user prompt and the agent descriptions
        best_score = best_agent['similarity']

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_prompt)


class ActionPlanningAgent:

    def __init__(self, openai_api_key, knowledge):
        # TODO: 1 - Initialize the agent attributes here
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        # TODO: 2 - Instantiate the OpenAI client using the provided API key
        client = OpenAI(api_key=self.openai_api_key,
                        base_url="https://openai.vocareum.com/v1")        
        # TODO: 3 - Call the OpenAI API to get a response from the "gpt-3.5-turbo" model.
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": f"""
                You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested
                to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. 
                Forget any previous context. 
                This is your knowledge: {self.knowledge}
                """},               
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )        
        response_text = response.choices[0].message.content  # TODO: 4 - Extract the response text from the OpenAI API response
        # TODO: 5 - Clean and format the extracted steps by removing empty lines and unwanted text
        steps = response_text.split("\n")
        # clean step output
        steps = [step.strip() for step in steps if step.strip() and step.strip()[0].isdigit()]
        return steps
