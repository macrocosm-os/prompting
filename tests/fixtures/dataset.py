
from prompting.tools import MockDataset, CodingDataset, WikiDataset, StackOverflowDataset, DateQADataset, MathDataset

DATASETS = [
    MockDataset,
    CodingDataset,
    WikiDataset,
    StackOverflowDataset,
    DateQADataset,
    MathDataset,
]

WIKI_ARTICLE = {
    'title': 'Emilio Alvarez (bishop)',
    'url': 'https://en.wikipedia.org/wiki/Emilio_Alvarez_(bishop)',
    'length': 8185,
    'extract': '<p><b>Emilio Alvarez</b> (born January 16) is a religious leader in the United States, and founding bishop of the Union of Charismatic Orthodox Churches. He is also the founding director of the Institute for Paleo-Orthodox Christian Studies (formerly the certificate in Convergence Studies Program at New York Theological Seminary).', 
    'backlinks': 7,
    'categories': [
        '21st-century American bishops',
        '21st-century Puerto Rican peopl',
        'nvergence Movemen',
        'Living peopl',
        'People of Afro–Puerto Rican descen',
        'Puerto Rican bishops',
        'Religious leaders from New York (state)',
        'Short description matches Wikid',
        'Writers from New York (state)',
        'Year of birth missing (living people)'
        ]
 }
