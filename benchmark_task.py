"""
Template for a task development experiment. Requires the following defintions:
- Dataset
- Task
- Challenge definition

Rewarding is not currently implemented.

Supports miner completions, but at present they musy be HuggingFaceLLM-compatible (and there is no control over their model kwargs)

Ultimately this should be replaced with a (pseudo) neuron and we should be using the 'run_step' method for a completely realistic simulation.

"""
import re
import tqdm
import time
import wandb
import torch
import random
import argparse
import functools
import pandas as pd
import bittensor as bt

from prompting.tasks import Task

from prompting.tools import Dataset
from prompting.agent import HumanAgent
from prompting.llm import load_pipeline, HuggingFaceLLM

####################################################################################################
# Here we define the dataset, task, and system prompts for the task development experiment

class ReviewDataset(Dataset):

    SENTIMENTS = ['positive','neutral','negative']
    # TODO: filter nonsense combinations of params

    query_template = 'Create a {style} review of a {topic} in the style of {mood} person in a {tone} tone. The review must be of {sentiment} sentiment.'
    params = dict(
        style = ['short','long','medium length','twitter','amazon','terribly written','hilarious'],
        mood = ['angry','sad','amused','bored','indifferent','shocked','terse'],
        tone = ['casual','basic','silly','random','thoughtful','serious','rushed'],
        topic = ['movie','book','restaurant','hotel','product','service','car','company','live event'],
        sentiment = SENTIMENTS
    )
    @property
    def size(self):
        return functools.reduce(lambda x, y: x * y, [len(v) for v in self.params.values()], 1)

    def __repr__(self):
        return f'{self.__class__.__name__} with template: {self.query_template!r} and {self.size} possible phrases'

    def random(self, *args, **kwargs):
        
        selected = {k: random.choice(v) for k,v in self.params.items()}
        links_unused = list(selected.values())
        return {
            'title': f'A {selected["sentiment"]} review of a {selected["topic"]}',
            'topic': selected['topic'],
            'subtopic': selected['sentiment'],
            'content': self.query_template.format(**selected),
            'internal_links': links_unused,
            'external_links': links_unused,    
            'source': self.__class__.__name__,       
        }
    
    def search(self, *args, **kwargs):
        return self.random()
    
    def get(self, *args, **kwargs):
        return self.random()


class SentimentAnalysisTask(Task):
    
    name = "sentiment analysis"
    desc = "get help analyzing the sentiment of a review"
    goal = "to get the sentiment to the following review"
    reward_definition = [
        dict(name="sentiment", weight=1.0),
    ]
    penalty_definition = []
    cleaning_pipeline = []

    static_reference = True
    query_system_prompt = """You are an assistant that generates reviews based on user prompts. You follow all of the user instructions as well as you can. Make the reviews as realistic as possible. Your response contains only the review, nothing more, nothing less"""
    
    def __init__(self, llm_pipeline, context, create_reference=True):

        self.context = context

        self.query_prompt = context.content
        self.query = self.generate_query(llm_pipeline)
        self.reference = context.subtopic

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags



####################################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description='Run a task benchmarking experiment')
    parser.add_argument('--mock', action='store_true', help='Run with a mock validator')
    parser.add_argument('--mock_output', default='This is a mock LLM output', help='Mock LLM output')  
    parser.add_argument('--model_id', default='HuggingFaceH4/zephyr-7b-beta', help='Which LLM to use for the experiment')
    parser.add_argument('--miners', nargs='+', default=[], help='Which miners to use')
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", help='Run on a specific device')
    parser.add_argument('--name', default='test-experiment', help='Experiment name')
    parser.add_argument('--generate_challenge', action='store_true', help='Use agent to generate challenge')
    parser.add_argument('--challenge_template', type=str, default=None, help='Challenge template string (to be used instead of agent LLM generation)')
    parser.add_argument('--reward', action='store_true', help='Enable rewarding of responses (miners must be set to take effect)')
    parser.add_argument('--wandb_off', action='store_true', help='Turn off wandb logging')
    parser.add_argument('--num_trials', type=int, default=1000, help='Number of trials to run')
    parser.add_argument('--num_completions', type=int, default=1, help='Number of completions to generate per miner')
    parser.add_argument('--miner_system_prompt', default="""You are a helpful assistant that always follows user prompts. Your responses only contain the requested information, without any greetings or friendly chit chat. You are straight to the point.""")
    parser.add_argument('--task_system_prompt', default=SentimentAnalysisTask.query_system_prompt)
    return parser.parse_args()

args = parse_args()
bt.logging.info(f'Args: {args}')


####################################################################################################
def get_completions(miners, challenge):
    # Generate completions
    
    completions = []
    for miner_name, miner in miners.items():
        for j in range(args.num_completions):
            t0 = time.time()
            completion = miner.query(challenge)
            completions.append({
                'completion_index': j,
                'completion_time': time.time() - t0,
                'miner_name': miner_name,
                'completion': completion
            })    
    return completions

def get_challenge(agent):
    # Generate a challenge, either using agent, a format string or the raw query
    
    if args.generate_challenge:
        agent.create_challene()
        return agent.challenge
    
    if args.challenge_template:
        expected_fields = re.findall(r'\{(\w+)\}', args.challenge_template)
        missing_fields = [field for field in expected_fields if not hasattr(agent.task, field)]
        assert not missing_fields, f'The following fields were specified in the challenge template but are not in the task ({missing_fields}). Did you use the correct names?'
        
        fields = {fields: getattr(agent.task, fields) for fields in expected_fields}
        challenge = args.challenge_template.format(**fields)
    else:
        challenge = agent.task.query
    
    agent.challenge = challenge
    agent.challenge_time = 0
    return challenge
                

def main(name, dataset, task_class):

    llm_pipeline = load_pipeline(model_id=args.model_id if not args.mock else args.mock_output, mock=args.mock, device=args.device)

    miners = {}
    if args.model_id in args.miners:
        miners[args.model_id] = HuggingFaceLLM(llm_pipeline, system_prompt=args.miner_system_prompt)
        args.miners.remove(args.model_id)

    miners.update({name: load_pipeline(model_id=name, mock=args.mock, device=args.device) for name in args.miners})
    bt.logging.info(f'Activated the following {len(miners)} miners: {miners}')
    
    if not args.generate_challenge and not args.challenge_template and args.miners:
        bt.logging.warning(f'The raw query will be used to get miner completions. This is not recommended as the question may be ill-posed and produce bad completions.')

    save_path = f'{name}.csv'
    results = []
    for i in tqdm.tqdm(range(args.num_trials), total=args.num_trials, desc='Running trials'):

        context = dataset.next()
        task = task_class(llm_pipeline, context)
        
        agent = HumanAgent(task=task, llm_pipeline=llm_pipeline, begin_conversation=False)
        challenge = get_challenge(agent)
            
        result = {
            'step': i,
            **agent.__state_dict__(full=True),
        }
        batch_results = []
        if miners:
            for completion in get_completions(miners, challenge):
                batch_results.append({**result, **completion})
        else:
            batch_results.append(result)                    

        results.extend(batch_results)
        
        if i%10 == 0 or i==args.num_trials-1:
            bt.logging.info(f'Saving results at step {i} to {save_path!r}')
            pd.DataFrame(results).to_csv(save_path, index=False)

        del task
        del agent
        
        if args.wandb_off:
            continue

        for batch_result in batch_results:
            wandb.log(batch_result)
    
    bt.logging.success(f'Experiment {name} complete after {args.num_trials} trials')

if __name__ == '__main__':

    ds = ReviewDataset()
    task_class = SentimentAnalysisTask
    arg_hash = hash(str(vars(args)))    
    name = f'{task_class.__name__}{arg_hash}'

    if not args.wandb_off:
        wandb.init(project='sn1-task-dev', name=name, config=args, save_code=True)

    main(name, dataset=ds, task_class=task_class)

