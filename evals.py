
import os
import re
import glob
import tqdm
import wandb
import random
import argparse
import pandas as pd
import bittensor as bt
import plotly.express as px

from openai import OpenAI
from functools import lru_cache
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(timeout=30, api_key=api_key)

api = wandb.Api()

####################################################################################################

DEFAULT_SYSTEM_PROMPT =  """
You are an AI assistant which judges the quality of other LLM generations.

The user will provide you with a system prompt, a user prompt and an LLM completion. Your task is to evaluate the quality of the completion given the user prompt. You can use the following scale: 1 - bad, 2 - ok, 3 - good, 4 - excellent. An excellent completion is one which follows the insruction perfectly. This means: it contains only the requested information and nothing more, it is coherent, convincing and well-written without any unrequested placeholders, tags or otherwise discursive and off topic language or content.

Do not provide any information other than the quality of the completion, expressed as a number between 1 and 4.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Run a set of gpt evals using a wandb run or local data')
    parser.add_argument('--path', type=str, help='Wandb or local filepath (glob) to get data from')
    parser.add_argument('--mock', action='store_true', help='Run with a mock validator')

    parser.add_argument('--tasks', type=str, nargs="+", default=None, help='Which tasks to evaluate. If not set, all the template strings will be applied to all the tasks in the data.')
    parser.add_argument('--templates', type=str, nargs="+", default=None, help='User template strings, to be used to form the user prompt')
    parser.add_argument('--template_names', type=str, nargs="+", default=None, help='User template names, to be used to name the evals')

    parser.add_argument('--wandb_off', action='store_true', help='Turn off wandb logging')
    parser.add_argument('--num_evals', type=int, default=5, help='Number of evals to run on each example')
    parser.add_argument('--max_samples', type=int, default=100, help='Max. number of samples to evaluate')

    parser.add_argument('--system_prompt', default=DEFAULT_SYSTEM_PROMPT)

    parser.add_argument('--model_id', default="gpt-4-turbo-preview", help='Which gpt to use for the eval')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature of gpt completion')
    parser.add_argument('--max_tokens', type=int, default=256, help='Max tokens of gpt completion')
    parser.add_argument('--seed', type=int, default=123, help='Random seed of gpt completion')

    return parser.parse_args()

args = parse_args()
bt.logging.info(f'Args: {args}')


def load_data(path):

    local_files = glob.glob(path)
    if local_files:
        bt.logging.info(f'Loading {len(local_files)} local files:\n{local_files}')
        frames = {}
        for path in local_files:
            frames[path] = pd.read_csv(path)

        df =  pd.concat(frames, keys=frames.keys()).reset_index()

    else:
        bt.logging.info(f'Loading wandb run: {path}')
        run = api.run(path)
        df = pd.DataFrame(list(run.scan_history()))
        df._timestamp = df._timestamp.apply(pd.to_datetime, unit='s')
        df['elapsed'] = df._timestamp.diff().dt.total_seconds()
        df['network_time'] = df.timings.apply(max)

    bt.logging.success(f'+ Loaded {len(df)} records')

    df['task'] = df['task'].apply(lambda x: re.sub(r'\W', '-', x))
    return df


@lru_cache(maxsize=16000)
def gpt_inference(message, system_prompt, engine, n=1, temperature=0.7, seed=1234, max_tokens=128):
    """Inference gpt
    TODO: allow args to be set
    """
    if args.mock:
        return random.choices(range(1,5), k=n)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
        ]
    client_response = client.chat.completions.create(
            messages=messages,
            model=engine,
            seed=seed,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    texts = [text.message.content.strip() for text in client_response.choices]
    return texts

def analyze_results(df, name, write=False):

    fig = px.histogram(df, x='eval_name', color='eval_result', facet_col='task',
            title=f'GPT Evals on {list(df.task.unique())} Task',
            barnorm='percent', opacity=0.7,
            category_orders={'eval_result': ['1', '2', '3', '4']},
            labels={'eval_result': 'Eval Score', 'task': 'Task', 'percent': '% of Generations'},
            width=800, height=600, template='plotly_white')

    fig.show()
    if write:
        if not os.path.exists(f'evals/'):
            os.makedirs(f'evals/')
        fig.write_image(f'evals/{name}.png')
    if not args.wandb_off:
        wandb.log({f'evals_histogram': wandb.Plotly(fig)})

def main(df, name):

    save_path = f'{name}.csv'
    results = []

    df = df.iloc[:args.max_samples].reset_index()
    pbar = tqdm.tqdm(df.iterrows(), total=len(df), desc='Evaluating data')
    for idx, row in pbar:
        for eval_name, template in zip(args.template_names, args.templates):

            expected_fields = re.findall(r'\{(\w+)\}', template)
            missing_fields = [field for field in expected_fields if field not in row]
            assert not missing_fields, f'The following fields were specified in template {eval_name} but are not in the data at row {idx} ({missing_fields}). Did you use the correct names?'

            fields = {field: row[field] for field in expected_fields}

            user_prompt = template.format(**fields)

            eval_results = gpt_inference(message = user_prompt, system_prompt = args.system_prompt, engine=args.model_id,
                                         temperature = args.temperature, n = args.num_evals,
                                         max_tokens = args.max_tokens, seed = args.seed
                                         )

            for k, eval_result in enumerate(eval_results):

                results.append({
                    'index': idx,
                    'task': row.task,
                    'eval_index': k,
                    'eval_name': eval_name,
                    'eval_result': eval_result,
                    'user_prompt': user_prompt,
                    'topic': row.topic,
                    'subtopic': row.subtopic,
                    **fields
                })
                if not args.wandb_off:
                    wandb.log(results[-1])

        if idx%10 == 0 or idx==len(df)-1:
            bt.logging.info(f'Saving results at step {idx} to {save_path!r}')
            pd.DataFrame(results).to_csv(save_path, index=False)

    df_results = pd.DataFrame(results)
    stats = df_results.groupby("task").eval_result.value_counts(dropna=False)
    bt.logging.success(f'Completed {len(df_results)} evals. Results\n{stats}')

    analyze_results(df_results, name)


if __name__ == '__main__':

    df = load_data(args.path)
    if not args.tasks:
        args.tasks = list(df.task.unique())
        bt.logging.warning(f'Setting {args.tasks=} manually using the available tasks in the data')

    arg_hash = hash(str(vars(args)))
    tasks = '-'.join(args.tasks)
    name = f'evals-{tasks}-{arg_hash}'

    if not args.wandb_off:
        wandb.init(project='sn1-task-evals', name=name, config=args, save_code=True)

    df_task = df.loc[df.task.isin(args.tasks)]
    if df_task.shape[0] == 0:
        raise ValueError(f'No samples were found in dataset which match task {args.tasks}. Options are {set(df.task.unique())}')

    bt.logging.success(f'Data contains {len(df)} samples which match tasks {args.tasks}')

    if not args.template_names:
        args.template_names = [f'eval-{i}' for i in range(len(args.templates))]
    elif args.template_names and len(args.template_names) == len(args.templates):
        raise ValueError(f'Template names are a different length to templates. {len(args.templates)=} and {len(args.template_names)=}')

    main(df_task, name)
