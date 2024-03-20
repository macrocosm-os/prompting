import functools
import random

from typing import Dict, Union, List, Tuple

from .base import Dataset
from ..selector import Selector


class GenericQuestionDataset(Dataset):

    # TODO: filter nonsense combinations of params
    
    query_template = 'Ask a {question_type} question about a {a} {b} related to {topic}'
    params = dict(
        question_type = ['casual','basic','silly','random','thoughtful','detailed','deep','fun'],
        a = ['surprising','controvesial', 'historic', 'famous', 'imfamous', 'popular', 'unpopular'],
        b = ['person','figure','opinion','event', 'leader', 'spokesperson','expert','topic'],
        # c = ['modern','ancient','current','world'],
        topic = ['science','politics','parenting','travel','cuisine','sports','pop culture','tech','history'],
    )    
    @property
    def size(self):
        return functools.reduce(lambda x, y: x * y, [len(v) for v in self.params.values()], 1)

    def __repr__(self):
        return f'{self.__class__.__name__} with template: {self.query_template!r} and {self.size} possible phrases'
    
    def next(self):
        selected = {k: random.choice(v) for k,v in self.params.items()}
        return {
            'query': self.query_template.format(**selected),
            'topic' : selected['topic'],
            'subtopic' : selected['question_type'],
            'tags' : [selected['a'], selected['b']]

            }
    
    def get(
        self,
        question_type: str,
        a: str,
        b: str,
        topic: str,
    ) -> Dict:
        """Gets a generic instruction query by passying in a question_type, a, b, and topic
        
        Args:
            question_type (str): the type of question
            a (str): the first part of the question
            b (str): the second part of the question
            topic (str): the topic of the question
            
        Returns:
            Dict: a dictionary with the query and the parameters"""
        
        return {
            'query': self.query_template.format(question_type=question_type, a=a, b=b, topic=topic),
            'topic' : topic,
            'subtopic' : question_type,
            'tags' : [a, b]
        }
    
    def get_random(self) -> Dict:
        """Gets a random generic instruction query
        
        Returns:
            Dict: a dictionary with the query and the parameters"""
        
        return self.next()