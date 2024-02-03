# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import random
import mathgenerator
import bittensor as bt
from sympy.parsing.latex import parse_latex

from .base import Dataset
from ..selector import Selector

class MathDataset(Dataset):
    topics_list = mathgenerator.getGenList()

    def __init__(self, seed=None):

        self.seed = seed
        self.rng = random.Random(seed)

    def random_problem(self, parse):
        if parse:
            parseable_list = [
                2,
                7,
                11,
                15,
                19,
                21,
                24,
                27,
                29,
                30,
                32,
                33,
                35,
                36,
                42,
                45,
                48,
                49,
                52,
                59,
                60,
                64,
                66,
                67,
                68,
                69,
                70,
                73,
                76,
                78,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                92,
                94,
                95,
                96,
                97,
                105,
                108,
                109,
                111,
                115,
                122,
                123,
            ]
            options = parseable_list
            choice = self.rng.choice((options))
            # TODO: When the solution contains the symbol x we should specify the x value and substitute it in the solution
            problem, solution = mathgenerator.genById(choice)
            _, subtopic, _, _, topic, _ = self.topics_list[choice]

            subs = {}
            # check if solution contains letters
            if "x" in solution:
                subs["x"] = 10
                bt.logging.warning(
                    "Coercing a symbolic expression to a numeric expression by substituting x=10"
                )

            # BUG: parse latex assumes that all letters are variables and so solutions like $No$ are interpreted as 'N * o'
            solution_numeric = parse_latex(
                str(solution).replace("$", "").strip()
            ).evalf(subs=subs)
            return {
                "problem": problem,
                "solution": solution_numeric,
                "solution_raw": solution,
                "topic": topic,
                "subtopic": subtopic,
            }
        else:
            options = mathgenerator.getGenList()
            choice = self.rng.choice(range(len(options)))
            problem, solution = mathgenerator.genById(choice)
            _, subtopic, _, _, topic, _ = self.topics_list[choice]
            return {
                "problem": problem,
                "solution": solution,
                "topic": topic,
                "subtopic": subtopic,
            }

    def next(self, parse=True):
        t0 = time.time()
        info = self.random_problem(parse)
        info["fetch_time"] = time.time() - t0
        return info
