import random
import pandas as pd
import bittensor as bt
from prompting.utils.exceptions import MaxRetryError

from typing import List

from prompting.tools import (
    Context,
    WikiDataset,
    HFCodingDataset,
    MathDataset,
    WikiDateDataset,
    MockDataset
)       

class TransitionMatrix:

    def __init__(self, labels: List[str], probs: List[List[float]], state: str=None, seed: int=None, begin_probs: List[float]=None, end_probs: List[float]=None):
        """Initializes a Markov transition matrix.

        Args:
            labels (List[str]): Name of states.
            probs (List[List[float]]): Probabilities of transitioning from one state to another.
            state (str, optional): Starting state. Defaults to None.
            seed (int, optional): Random seed. Defaults to None.
            begin_probs (List[float], optional): Probabilities of starting in each state. Defaults to None.
            end_probs (List[float], optional): Probabilities of ending in each state. Defaults to None.
        """

        self.labels = labels
        self.matrix = pd.DataFrame(probs, columns=labels, index=labels)

        self.rng = random.Random(seed)

        self.state = state or self.rng.choices(self.labels, weights=begin_probs, k=1)[0]
        self.begin_probs = begin_probs
        self.end_probs = end_probs
        self.history = [self.state]

    def next(self, last=False) -> str:
        """Selects the next task probabilistically based on the current task and the Markov transition matrix."""

        probs = None
        if last and self.end_probs is not None:
            probs = self.end_probs
        else:
            probs = self.matrix.loc[self.state].to_list()

        self.state = self.rng.choices(self.matrix.columns, weights=probs, k=1)[0]
        self.history.append(self.state)
        return self.state

    def random(self):
        return self.rng.choices(self.labels, k=1)[0]



class ContextChain:

    # Chain rule for transitioning between tasks, this specifies the method to use to get a new context when transitioning to a new task
    # internal specifies the dataset method to use when choosing the same task again
    # external specifies the dataset method to use when 'arriving' at a new task
    CHAIN_RULE = {
        'qa': {
            'internal': 'get', # use same page and a different section
            'external': 'search', # search for a new page when transitioning to QA
            'dataset': WikiDataset(),
            },
        'summarization': {
            'internal': 'get',
            'external': 'search',
            'dataset': WikiDataset()
            },
        'date_qa': {
            'internal': 'random', # when creating a new context, use a random date as the seed
            'external': 'random', # there's no external link to use when transitioning to DateQA - we need a valid date
            'dataset': WikiDateDataset()
            },
        'debugging': {
            'internal': 'random',
            'external': 'random',
            'dataset': HFCodingDataset()
        },
        'math': {
            'internal': 'get',
            'external': 'search',
            'dataset': MathDataset()
            },
    }

    @property
    def task_name_history(self):
        return [h['task'] for h in self.history]

    def __init__(self, matrix: TransitionMatrix, num_steps=5, seed=None, continuity=1, coherence=1, start_params=None, mock=False, max_tries=10):

        self.matrix = matrix
        self.task_name = matrix.state

        self.num_steps = num_steps
        self.continuity = continuity
        self.coherence = coherence

        self.rng = random.Random(seed)
        self.selector = random.choice

        self.step = 0
        self.context = None
        self.history = []
        self.start_params = start_params or {}

        self.mock = mock
        self.max_tries = max_tries


    def walk(self):
        return self.__next__()

    def __iter__(self):
        return self

    def _get_context(self, prev_task: str, next_task: str) -> Context:
        """Creates a coherent context based on the previous task, the next task and the current context.

        Args:
            prev_task (str): Previous task name.
            next_task (str): Next task name.

        Returns:
            Context: A dataset context.
        """

        change_task = prev_task != next_task
        spec = self.CHAIN_RULE[next_task]

        params = {}
        if self.step == 0:
            # If this is the first step, use the start_params to get the first context or use a random context
            params = self.start_params or {'method': 'random'}

        elif change_task:
            # If we are changing tasks, we select an external link from the current context and use it to seed the next context
            links = self.context.external_links
            link = self.rng.choice(links)
            bt.logging.info(f'Selected external link for new page: {link!r} from {len(links)} links {links[:5]}...')
            params.update({'name':link, 'method':spec['external']})

        else:
            params['method'] = method = spec['internal']

            # stay on same page and choose a new section
            # TODO: make this based on selectors and configurable
            if self.rng.random() > 0.7:
                params.update({'name':self.context.title, 'exclude':[self.context.topic]})
                bt.logging.info(f'Staying on the same page {self.context.title!r} and choosing a new section... excluding={params["exclude"]}')
            else:
                links = self.context.external_links
                link = self.rng.choice(links)
                bt.logging.info(f'Selected link: {link!r} from {len(links)} links {links[:5]}...')
                params.update({'name':link})

        dataset = spec['dataset'] if not self.mock else MockDataset()

        bt.logging.info(f'Transitioning to {next_task!r}, using dataset {dataset.__class__.__name__}.next({params})')
        return dataset.next(**params)

    def _already_in_history(self, context, fields=('title', 'topic', 'subtopic')):
        """Check if a similar context is already in the history. We define similarity as having the same title, topic and subtopic."""

        same_fields = lambda x: all(getattr(x,f) == getattr(context,f) for f in fields)
        return any(same_fields(h['context']) and context['task']==h['task'] for h in self.history)

    def __next__(self):
        if self.step == self.num_steps:
            raise StopIteration

        prev_task = self.task_name

        # uses Markov chain to select the next task probabilistically
        last = self.step == self.num_steps - 1
        next_task = self.matrix.next(last=last) if self.step > 0 else self.matrix.state

        tries = 0
        # Keep trying to create a new task and context
        while True:

            tries += 1
            if tries > self.max_tries:
                raise MaxRetryError(f'Could not find a valid next task after {self.max_tries} tries...')
            try:
                # TODO: What if task is fundamentally incompatible with the previous task? We just time out?
                context = self._get_context(prev_task, next_task)
                if self._already_in_history(context):
                    bt.logging.error(f'Context {context} already in history...')
                    continue
                break
            except Exception as e:
                bt.logging.error(e)

        self.task_name = next_task
        self.context = context
        bt.logging.success(f'Context: {context}')

        self.step += 1
        self.history.append({'task': self.task_name, 'context': context})

        return context

    def __len__(self):
        return self.num_steps

if __name__ == '__main__':

    labels = ['QA', 'Summarization', 'DateQA', 'Debugging', 'Mathematics']

    probs =  [
        [0.6, 0.35, 0.15, 0.0, 0.0],  # QA
        [0.65, 0.3, 0.05, 0.0, 0.0],  # Summarization
        [0.4, 0.2, 0.4, 0.0, 0.0],  # DateQA
        [0.3, 0.2, 0.0, 0.5, 0.0],  # Debugging
        [0.3, 0.2, 0.0, 0.0, 0.5],  # Mathematics
    ]
    mat = TransitionMatrix(labels=labels, probs=probs, seed=42)
    num_steps = 5
    for j in range(num_steps-1):
        mat.next(last=j == num_steps-1)
    print(mat.history)

    # Reset the matrix (but keep the seed)
    matrix = TransitionMatrix(labels=labels, probs=probs, seed=42)
    chain = ContextChain(matrix=matrix, num_steps=num_steps, seed=None, mock=False)

    for i,context in enumerate(chain, start=1):
        task_name = chain.task_name
        print(f'Step {i}. {task_name:<14}: {context}')
