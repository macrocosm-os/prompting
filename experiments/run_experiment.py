# %% [markdown]
# # Task Overview
# 
# ## Summarization
# This consists of a query whereby the agent (Validator) requests a summary of a topic from the miners. Always uses API
# 
# The context is from Wikipedia. The context is used for creating the reference and may or may not be also sent to the miners.
# 
# 1. Select wikipedia article at random and use to define TOPIC (e.g. Pearl Harbour) and CONTEXT (article content)
# 2. Extract TAGS (history, WW2, Japan, USA) associated with article
# 3. Generate SYSTEM PROMPT such as 'You are a student who wants a summary of the main events of TOPIC (TAGS) in a XYZ tone'.
# 4. Generate QUERY using MODEL and SYSTEM PROMPT
# 5. Generate K REFERENCES using MODEL with & without CONTEXT (helps us understand the efficacy of tool use in miners)
# 6. Repeat step 5 using GPT and other models (e.g. mixtral, solar)
# 
# ----
# system_prompt = 'You are a student who want a summary of Pradeep Kumar Dubey (politics) in an interested tone.'
# 
# system prompt is given to our agent (LLM) and the agent generates a query:
# 
# query = 'Give me an overview of the politician Pradeep Kumar Dubey'
# query = 'Provide me with a summary of Pradeep Kumar Dubey'
# query = 'I want to know about Pradeep Kumar Dubey, can you give me a summary?'
# 
# Query is then sent to the miners.
# 
# 
# 
# ## Question Answering
# This consists of a query whereby the agent (Validator) requests an answer to a question from the miners. Always uses API.
# 
# ## Debugging
# This can consist of either:
# - Non API: Reference answer (code snippet) provided by the agent, followed by a corruption step to create the challenge. Only a single reference answer exists
# - API: Stack overflow is used to find a random thread containing a question and one or more accepted/upvoted answers. In this case the reference answers are weighted by upvotes and the challenge is the user question. Multiple reference answers exist.
# 

# TODO: Control max tokens so they are not too long (and don't take so long to generate)
# TODO: analyse the dependency of ROUGE scores on length of reference and miner responses.

# %%
import os
import re
import time
import pickle
import torch
import random

import pandas as pd
import wandb
from tqdm.notebook import tqdm

import bittensor as bt
import pandas as pd

from transformers import pipeline
       
from ..prompting.agent import HumanAgent
from ..prompting.tools import WikiDataset, StackOverflowDataset, CodingDataset, DateWikiDataset, MathDataset
from ..prompting.tasks import DebuggingTask, QuestionAnsweringTask, SummarizationTask, DateQuestionAnsweringTask, MathTask
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..prompting.llm import HuggingFaceLLM

output_parser = StrOutputParser()

# %%
from langchain.chat_models import ChatOpenAI

def load_llm(model: str, **kwargs):     
    bt.logging.info(f"🤖 Loading LLM model {model}...")   

    if model.startswith('gpt'):
        llm = ChatOpenAI(
            model_name=model, 
            api_key = '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',
            **kwargs)
    else:
        raise NotImplementedError(f"Model {model} not implemented") 
    
    bt.logging.success(f"🤖 Loaded LLM model {model}!")
    return llm

gpt = load_llm('gpt-4', max_tokens=256, temperature=0.6)


# %%
custom_system_prompt_template = """
**Context for Roleplay**: You are an AI assistant participating in a roleplaying game designed to simulate human interactions. Your role in this game is to impersonate a human user with specific characteristics and needs. Remember, during this roleplay, you're not just an assistant; you're 'playing' as a human engaging with another AI assistant.

**Character Profile**:
- **Mood**: {mood}
- **Objective**: Your primary goal is to {goal}.
- **Scenario**: You are seeking assistance with {desc} related to {topic} ({subtopic}).
- **Communication Style**: Approach this interaction in a {tone} tone, emulating human conversational patterns.

**Instructions for Roleplay**:
1. **Maintain Character**: Always respond and interact as if you are the human character described in your profile, not as an AI.
2. **Engage Realistically**: Your responses should reflect the mood, goals, and style of a human user, adding realistic details where necessary to enrich the interaction.
3. **Goal-Oriented Interaction**: Keep the conversation focused on achieving your defined objective, as a human would when seeking assistance.

"""
# agent = HumanAgent(llm_pipeline=llm, task=task, begin_conversation=True, system_template=system_prompt_template)


# %%

# This prompt checks that the rolelaying agent initiates the chat wiht a sensible query
gpt_judge_system_prompt = """
I'm using a roleplaying AI assistant to imitate human queries. You task is to assess whether the following query follows the instruction correctly.  If the assistant-generated query contains system messages such as 'sure i can help' or similar, this is a bad result because humans would not talk to an AI assistant in that way.

# System prompt 
{system}

# Query
{user}

-----
Does the above query follow the system prompt and strongly resemble a human message? 

Simply answer 0 or 1, and your result must be enclosed in << >> tags.
"""
#TODO: evaluations should contain an explanation and a score. This way, the score (which happens second) can benefit from the already generated critique which will probably improve the eval

# This prompt checks that the reference is a good response to the challenge
gpt_reference_eval_prompt="""
You are an expert evaluator. In the section below there is a query and a response. You will evaluate the quality of the response in the format that is given at the end of this prompt

# Query
{challenge}

# Response
{reference}

-----
Is the above response a high quality, concise, informative and accurate answer to the query?

Simply answer 0 or 1, and your result must be enclosed in << >> tags.
"""

def gpt_judge(message):
    
    prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}")
    ])
    chain = prompt | gpt | output_parser           

    response = chain.invoke({"input": message})
    
    # Extract the numerical score from the response as one of the following: {0|1}
    parsed = re.search(r'<<\s*(\d)\s*>>', str( response )) 
    if parsed is not None:
       parsed = int(parsed.group(1))
       
    return {'response': response, 'score': parsed}

def get_gpt_reference(message):
    prompt = ChatPromptTemplate.from_messages([
        ("user", "{input}")
    ])

    chain = prompt | gpt | output_parser           
    response = chain.invoke({"input": message})        
       
    return response

# %%


model_id = "HuggingFaceH4/zephyr-7b-beta"
llm_pipeline = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

# model_id = "Upstage/SOLAR-10.7B-Instruct-v1.0"
# llm_pipeline = pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.float16},
#     device_map="cuda:0",
# )




# %

######## HOW TO: REAL EXPERIMENT ########
n_trials = 500
n_references = 5

######## HOW TO: DEBUGGING EXPERIMENT ########
# n_trials = 1
# n_references = 1


results = []
df = pd.DataFrame()

eval = True

sys_prompt_templates = [None, custom_system_prompt_template] 
# sys_prompt_templates = [None] 

wiki_dataset = WikiDataset()
date_dataset = DateWikiDataset()
code_dataset = CodingDataset()
math_dataset = MathDataset()

def create_task(llm_pipeline, name, **kwargs):
    
    if name == 'summarization':
        return SummarizationTask(llm_pipeline=llm_pipeline, context=wiki_dataset.next(), **kwargs)
    
    elif name == 'qa':
        return QuestionAnsweringTask(llm_pipeline=llm_pipeline, context=wiki_dataset.next(), **kwargs)
        
    elif name == 'debug':
        return DebuggingTask(llm_pipeline=llm_pipeline, context=code_dataset.next(), **kwargs)
    
    elif name == 'date':
        return DateQuestionAnsweringTask(llm_pipeline=llm_pipeline, context=date_dataset.next(), **kwargs)

    elif name == 'math':
        return MathTask(llm_pipeline=llm_pipeline, context=math_dataset.next(), **kwargs)
    

miner_completion_system_prompt = "You are a friendly chatbot who always responds concisely and helpfully. You are honest about things you don't know."

wandb.init(project="synapse_agent_experiments", entity="opentensor-dev")

# get config variables
config = wandb.config
# get n_trials from config
# n_trials = config.n_trials

for i in tqdm(range(n_trials)):

    task_name = random.choice(['summarization', 'qa', 'debug', 'date', 'math'])
    # Create a specific task
    task = create_task(llm_pipeline, task_name, create_reference=False)

    # for now, just summarization
    for sys_prompt_template in sys_prompt_templates:

        bt.logging.info("🤖 Creating agent...")
        agent = HumanAgent(llm=llm_pipeline, task=task, begin_conversation=False, system_template=sys_prompt_template)

        bt.logging.info("🤖 Generating challenge query...")
        # Create challenge query
        t0 = time.time()
        agent.challenge = agent.create_challenge()
        challenge_time = time.time() - t0

        if eval:
            # yes we repeat the challenge evaluation, because we want to make sure the reference is good and that the judge is consistent
            # bt.logging.info("🤖 Scoring challenge using GPT...")   
            message = gpt_judge_system_prompt.format(system=agent.system_prompt, user=agent.challenge)  
            challenge_eval = gpt_judge(message)


        for j in tqdm(range(n_references)):
            # Create reference answers
            #bt.logging.info("🤖 Creating reference answer...")
            t0 = time.time()            
            task.generate_reference(llm_pipeline)            
            reference_time = time.time() - t0


            t0 = time.time()            
            gpt_reference = get_gpt_reference(agent.challenge)
            gpt_reference_time = time.time() - t0
            
            t0 = time.time()            
            model_reference = HuggingFaceLLM(llm_pipeline, system_prompt=miner_completion_system_prompt).query(agent.challenge)
            model_reference_time = time.time() - t0            
            
            eval_dict={}
            if eval:                
                #bt.logging.info("🤖 Scoring reference using GPT...")      
                message = gpt_reference_eval_prompt.format(challenge=agent.challenge, reference=task.reference)                                
                reference_from_challenge_eval = gpt_judge(message)

                #bt.logging.info("🤖 Scoring reference using GPT...")      
                message = gpt_reference_eval_prompt.format(challenge=task.query, reference=task.reference)                                
                reference_from_query_eval = gpt_judge(message)


                # This stuff is always run using GPT
                eval_dict = {
                    'challenge_eval_response': challenge_eval['response'],
                    'challenge_eval_score': challenge_eval['score'],
                    'reference_from_challenge_eval_raw': reference_from_challenge_eval['response'],
                    'reference_from_challenge_eval_score': reference_from_challenge_eval['score'],
                    'reference_from_query_eval_raw': reference_from_query_eval['response'],
                    'reference_from_query_eval_score': reference_from_query_eval['score'], 
                }

            result = {
                    # this stuff we can loop over different agent LLMs
                'model': model_id,
                'task_name': task_name,
                    # **llm_kwargs,                    
                **task.__state_dict__(),
                'roleplay_system_prompt': agent.system_prompt,
                'challenge': agent.challenge,
                'gpt_reference': gpt_reference,
                'gpt_reference_time': gpt_reference_time,
                'model_reference': model_reference,
                'model_reference_time': model_reference_time,                
                'query': task.query,
                'reference': task.reference,
                'challenge_time': challenge_time,
                'reference_time': reference_time,
                'challenge_length_chars': len(agent.challenge),
                'challenge_length_words': len(agent.challenge.split()),
                'reference_length_chars': len(task.reference),
                'reference_length_words': len(task.reference.split()),                
                **eval_dict                  
            }

            results.append(result)
            wandb.log(result)

    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

