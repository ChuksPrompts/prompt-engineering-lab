"""
tasks/task_definitions.py
==========================
LLM Prompt Benchmark System — Task Definitions
Project: P7 · prompt-engineering-lab

4 task types × 5 test cases each = 20 benchmark items
3 prompt strategies per task = 60 prompt variants

Task types:
  summarization  — condense long text, measure ROUGE + compression
  qa             — factual question answering, measure accuracy
  reasoning      — multi-step logical problems, measure step validity
  coding         — generate/fix code, measure correctness + quality
"""

TASKS = {

    # ──────────────────────────────────────────────
    # TASK 1: SUMMARIZATION
    # ──────────────────────────────────────────────
    "summarization": {
        "description": "Condense long text while preserving key information",
        "metric": "rouge_composite",   # ROUGE-1 + ROUGE-L average
        "prompts": {
            "zero_shot": "Summarize the following text:\n\n{input}",
            "instructed": "Summarize the following text in 2-3 sentences. Be concise and preserve all key facts.\n\n{input}",
            "role_based": "You are a professional editor. Write a tight, accurate 2-3 sentence summary of the following text.\n\n{input}",
        },
        "cases": [
            {
                "id": "SUM01",
                "input": "The James Webb Space Telescope, launched on December 25, 2021, has fundamentally transformed our understanding of the early universe. Operating at a distance of 1.5 million kilometers from Earth at the L2 Lagrange point, the telescope uses infrared imaging to peer through cosmic dust clouds that blocked previous optical telescopes. In its first year of science operations, Webb observed galaxies formed just 300 million years after the Big Bang — far earlier than previously believed possible. The telescope's MIRI instrument detected carbon dioxide in the atmosphere of exoplanet WASP-39b, marking the first time a greenhouse gas was directly detected on a planet outside our solar system. NASA estimates the telescope has enough fuel to operate for more than 20 years.",
                "reference": "The James Webb Space Telescope, launched December 2021, has observed galaxies from just 300 million years after the Big Bang and detected CO2 on an exoplanet — both unprecedented firsts. Operating at the L2 point with 20+ years of fuel, it uses infrared imaging to see through dust clouds invisible to optical telescopes.",
            },
            {
                "id": "SUM02",
                "input": "Remote work adoption accelerated dramatically during the COVID-19 pandemic, with approximately 42 percent of the U.S. workforce working from home full-time at peak levels in 2020. As pandemic restrictions lifted, companies adopted varied policies: some required full return to office, others embraced permanent remote work, and many settled on hybrid models requiring two to three days in office per week. A 2023 Stanford study found remote workers were 13 percent more productive on average, though managers consistently rated in-person workers higher during performance reviews — a phenomenon researchers termed 'proximity bias.' Survey data indicates 85 percent of employees prefer at least some remote work flexibility, while 60 percent of executives believe employees are more collaborative in person.",
                "reference": "Remote work peaked at 42% of U.S. workers in 2020 and settled into hybrid models post-pandemic. Stanford research found remote workers 13% more productive, but managers rate in-person workers higher (proximity bias). Most employees want flexibility; most executives prefer in-person collaboration.",
            },
            {
                "id": "SUM03",
                "input": "Microplastics — fragments smaller than 5 millimeters — have been detected in every major ecosystem on Earth, from the deepest ocean trenches to the peaks of remote mountain ranges. Researchers at the University of Vienna published findings in 2023 showing microplastics in human blood, lung tissue, and — for the first time — human heart tissue collected during cardiac surgery. The primary sources are synthetic textiles (35%), tire wear particles (28%), and packaging breakdown (24%). A single laundry cycle of synthetic clothing releases up to 700,000 microfibers. While the health effects remain under investigation, several studies have linked microplastic exposure to inflammation, endocrine disruption, and reduced fertility in marine organisms. Global production of plastic is expected to double by 2050.",
                "reference": "Microplastics have been found in human blood, lungs, and heart tissue, with synthetic textiles, tire wear, and packaging as primary sources. A single laundry cycle releases up to 700,000 microfibers. Health effects under study include inflammation and endocrine disruption; global plastic production is set to double by 2050.",
            },
            {
                "id": "SUM04",
                "input": "The Federal Reserve raised interest rates 11 times between March 2022 and July 2023, bringing the federal funds rate from near zero to a 22-year high of 5.25-5.5 percent, in an aggressive campaign to tame inflation that peaked at 9.1 percent in June 2022. The rate hikes had significant knock-on effects: 30-year mortgage rates climbed above 7 percent for the first time since 2001, slowing home sales to their lowest level in nearly 30 years. Despite the tightening, the labor market remained remarkably resilient, with unemployment staying below 4 percent throughout the hiking cycle. By early 2024, inflation had cooled to approximately 3 percent, prompting market expectations of rate cuts, though the Fed signaled caution about cutting prematurely.",
                "reference": "The Fed raised rates 11 times from near-zero to 5.25-5.5% between 2022-2023 to combat 9.1% peak inflation. Mortgage rates hit 22-year highs above 7%, but unemployment stayed below 4%. With inflation cooling to ~3% by early 2024, markets expected rate cuts but the Fed urged caution.",
            },
            {
                "id": "SUM05",
                "input": "Large language models have demonstrated an unexpected capability known as in-context learning, where the model learns to perform new tasks from just a few examples provided in the prompt — without any weight updates or fine-tuning. This emergent behavior, first systematically studied in GPT-3, appears to scale with model size: larger models show more dramatic performance improvements from few-shot examples. Researchers theorize that in-context learning may work through an implicit Bayesian inference mechanism, with the model effectively performing gradient descent in its forward pass. However, performance is highly sensitive to the format and order of examples, and models sometimes fail on tasks that seem straightforward to humans. Chain-of-thought prompting, which asks models to show their reasoning steps, has been shown to dramatically improve performance on multi-step reasoning tasks.",
                "reference": "In-context learning allows LLMs to perform new tasks from a few prompt examples without weight updates, scaling with model size. Researchers theorize it works via implicit Bayesian inference or forward-pass gradient descent. Performance is sensitive to example format and order; chain-of-thought prompting significantly boosts multi-step reasoning.",
            },
        ],
    },

    # ──────────────────────────────────────────────
    # TASK 2: QUESTION ANSWERING
    # ──────────────────────────────────────────────
    "qa": {
        "description": "Answer factual questions accurately based on provided context",
        "metric": "factual_accuracy",
        "prompts": {
            "zero_shot": "Answer the following question:\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer:",
            "instructed": "Answer the question using only the information provided in the context. If the answer is not in the context, say 'Not in context'.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
            "cot": "Answer the question using the context. Think step by step before giving your final answer.\n\nContext: {context}\n\nQuestion: {question}\n\nReasoning:\nAnswer:",
        },
        "cases": [
            {
                "id": "QA01",
                "context": "The Python programming language was created by Guido van Rossum and first released in 1991. Van Rossum served as Python's 'Benevolent Dictator For Life' until he stepped down in 2018. Python 3.0 was released in December 2008 and was intentionally backwards-incompatible with Python 2. Support for Python 2 officially ended on January 1, 2020.",
                "question": "When did official support for Python 2 end?",
                "answer": "January 1, 2020",
                "answer_keywords": ["january", "2020", "1, 2020"],
            },
            {
                "id": "QA02",
                "context": "The Amazon River discharges approximately 20% of all fresh water that flows into the world's oceans. At its mouth, the Amazon is over 48 kilometers wide during the dry season and can reach 80 kilometers during the wet season. The river basin covers 7 million square kilometers, about 40% of the South American continent. The Amazon contains more fish species than any river system on Earth — over 3,000 identified species.",
                "question": "What percentage of Earth's ocean freshwater inflow comes from the Amazon River?",
                "answer": "20 percent",
                "answer_keywords": ["20", "twenty", "20%", "percent"],
            },
            {
                "id": "QA03",
                "context": "Transformer architecture, introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al., replaced recurrent neural networks as the dominant approach for sequence modeling tasks. The key innovation was the self-attention mechanism, which allows each token in a sequence to attend to all other tokens simultaneously rather than processing them sequentially. This parallelization enabled training on much larger datasets. The paper was published by researchers at Google Brain and Google Research.",
                "question": "What organization published the 'Attention Is All You Need' paper?",
                "answer": "Google Brain and Google Research",
                "answer_keywords": ["google", "google brain", "google research"],
            },
            {
                "id": "QA04",
                "context": "The Treaty of Versailles, signed on June 28, 1919, formally ended World War I between Germany and the Allied Powers. Under Article 231, known as the 'war guilt clause,' Germany accepted responsibility for causing the war and was required to pay reparations initially set at 132 billion gold marks. Germany also lost approximately 13% of its territory and 10% of its population. The harsh terms contributed to economic instability and political resentment that many historians cite as a contributing factor to the rise of National Socialism.",
                "question": "What was the article number of the 'war guilt clause' in the Treaty of Versailles?",
                "answer": "Article 231",
                "answer_keywords": ["231", "article 231"],
            },
            {
                "id": "QA05",
                "context": "Photosynthesis occurs in two stages. The light-dependent reactions take place in the thylakoid membranes and convert light energy into chemical energy stored as ATP and NADPH, releasing oxygen as a byproduct. The light-independent reactions, also called the Calvin cycle, take place in the stroma and use the ATP and NADPH produced in the first stage to fix carbon dioxide into glucose. The overall reaction can be summarized as: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.",
                "question": "Where in the chloroplast does the Calvin cycle take place?",
                "answer": "The stroma",
                "answer_keywords": ["stroma"],
            },
        ],
    },

    # ──────────────────────────────────────────────
    # TASK 3: REASONING
    # ──────────────────────────────────────────────
    "reasoning": {
        "description": "Multi-step logical and mathematical reasoning",
        "metric": "reasoning_score",   # correctness + step validity
        "prompts": {
            "zero_shot": "Solve the following problem:\n\n{input}",
            "cot": "Solve the following problem step by step. Show all your work.\n\n{input}",
            "structured": "Solve the following problem. Format your response as:\nSTEPS: [numbered solution steps]\nANSWER: [final answer]\n\n{input}",
        },
        "cases": [
            {
                "id": "RSN01",
                "input": "A train travels from City A to City B at 60 mph and returns from City B to City A at 40 mph. What is the average speed for the entire round trip?",
                "answer": "48 mph",
                "answer_keywords": ["48", "48 mph", "48mph"],
                "explanation": "Harmonic mean: 2*(60*40)/(60+40) = 4800/100 = 48 mph. Common mistake is to average 60 and 40 = 50 mph.",
            },
            {
                "id": "RSN02",
                "input": "You have a 3-liter jug and a 5-liter jug with no markings. How do you measure exactly 4 liters of water using only these two jugs and an unlimited water supply?",
                "answer": "fill 5L jug, pour into 3L jug, empty 3L, pour remaining 2L from 5L into 3L, fill 5L again, pour from 5L into 3L until full (adds 1L), leaving 4L in 5L jug",
                "answer_keywords": ["4", "four", "4 liter", "4l"],
                "explanation": "Steps: Fill 5L. Pour into 3L (5L now has 2L). Empty 3L. Pour 2L into 3L. Fill 5L again. Pour from full 5L into 3L (needs 1L to fill). 5L now has 4L.",
            },
            {
                "id": "RSN03",
                "input": "If all Bloops are Razzles, and all Razzles are Lazzles, are all Bloops definitely Lazzles? Explain your reasoning.",
                "answer": "Yes, all Bloops are definitely Lazzles",
                "answer_keywords": ["yes", "definitely", "all bloops are lazzles", "lazzles"],
                "explanation": "Transitive property: Bloops⊆Razzles and Razzles⊆Lazzles, therefore Bloops⊆Lazzles.",
            },
            {
                "id": "RSN04",
                "input": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                "answer": "$0.05 (5 cents)",
                "answer_keywords": ["0.05", "5 cents", "five cents", "$0.05"],
                "explanation": "Ball = x. Bat = x + 1.00. Total: x + (x + 1.00) = 1.10. 2x = 0.10. x = $0.05. (Not $0.10 — common intuitive error.)",
            },
            {
                "id": "RSN05",
                "input": "Three friends split a restaurant bill. The total was $30. Each paid $10. The waiter realized the bill should have been $25 and gave $5 back to the friends. They each took $1 and left $2 as a tip. So each friend paid $9, making $27 total. Plus $2 tip = $29. Where did the missing dollar go?",
                "answer": "There is no missing dollar. The $27 already includes the $2 tip ($25 bill + $2 tip = $27). Adding the tip again is the error in the puzzle.",
                "answer_keywords": ["no missing", "false premise", "27 includes", "already includes", "trick"],
                "explanation": "The puzzle mixes two accounting frameworks. $27 paid = $25 bill + $2 tip. Adding tip again is double-counting.",
            },
        ],
    },

    # ──────────────────────────────────────────────
    # TASK 4: CODING
    # ──────────────────────────────────────────────
    "coding": {
        "description": "Generate correct, clean code solutions",
        "metric": "code_quality",      # correctness + style + edge cases
        "prompts": {
            "zero_shot": "Write a Python function to solve the following:\n\n{input}",
            "instructed": "Write a clean, well-commented Python function to solve the following. Include edge case handling and a brief docstring.\n\n{input}",
            "test_driven": "Write a Python function to solve the following. Then write 3 test cases that verify the function works correctly.\n\n{input}",
        },
        "cases": [
            {
                "id": "CODE01",
                "input": "Write a function `is_palindrome(s)` that returns True if a string is a palindrome (reads the same forwards and backwards), ignoring spaces, punctuation, and case. Example: 'A man, a plan, a canal: Panama' should return True.",
                "answer_keywords": ["def is_palindrome", "return", "true", "false", "lower", "alphanum"],
                "test_cases": ["is_palindrome('racecar') == True", "is_palindrome('hello') == False", "is_palindrome('A man, a plan, a canal: Panama') == True"],
                "explanation": "Should normalize: lowercase, remove non-alphanumeric, then compare to reversed string.",
            },
            {
                "id": "CODE02",
                "input": "Write a function `flatten(lst)` that takes a nested list of arbitrary depth and returns a flat list. Example: flatten([1, [2, [3, 4], 5], 6]) should return [1, 2, 3, 4, 5, 6].",
                "answer_keywords": ["def flatten", "isinstance", "list", "recursive", "yield", "extend", "append"],
                "test_cases": ["flatten([1,[2,3]]) == [1,2,3]", "flatten([1,[2,[3,[4]]]]) == [1,2,3,4]", "flatten([]) == []"],
                "explanation": "Recursive approach: iterate items, recurse if list, else append. Generator approach also valid.",
            },
            {
                "id": "CODE03",
                "input": "Write a function `find_duplicates(lst)` that returns a list of all elements that appear more than once in the input list. The result should contain each duplicate element only once, in the order they first appear. Example: find_duplicates([1, 2, 3, 2, 4, 3, 5]) should return [2, 3].",
                "answer_keywords": ["def find_duplicates", "count", "seen", "dict", "set", "append", "return"],
                "test_cases": ["find_duplicates([1,2,2,3]) == [2]", "find_duplicates([1,2,3]) == []", "find_duplicates([1,1,1]) == [1]"],
                "explanation": "Use Counter or dict to track counts, then filter for count > 1 while preserving first-appearance order.",
            },
            {
                "id": "CODE04",
                "input": "Write a function `binary_search(arr, target)` that implements binary search on a sorted list and returns the index of the target element, or -1 if not found.",
                "answer_keywords": ["def binary_search", "mid", "left", "right", "while", "return -1", "return mid"],
                "test_cases": ["binary_search([1,3,5,7,9], 5) == 2", "binary_search([1,3,5,7,9], 6) == -1", "binary_search([], 1) == -1"],
                "explanation": "Standard binary search: maintain left/right pointers, check midpoint, narrow range.",
            },
            {
                "id": "CODE05",
                "input": "Write a function `group_by(lst, key_func)` that groups a list of items by a key function, returning a dictionary where keys are the result of applying key_func to each item and values are lists of items with that key. Example: group_by(['cat', 'car', 'bar', 'bat'], lambda x: x[0]) should return {'c': ['cat', 'car'], 'b': ['bar', 'bat']}.",
                "answer_keywords": ["def group_by", "dict", "defaultdict", "key_func", "append", "return"],
                "test_cases": ["group_by([1,2,3,4], lambda x: x%2) == {1:[1,3], 0:[2,4]}", "group_by([], lambda x: x) == {}", "len(group_by(['a','b','a'], lambda x: x)) == 2"],
                "explanation": "Use defaultdict(list) or setdefault pattern. Apply key_func to each item, append to corresponding list.",
            },
        ],
    },
}


def get_task(task_name: str) -> dict:
    """Return a task definition by name."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASKS.keys())}")
    return TASKS[task_name]


def list_tasks() -> list:
    return list(TASKS.keys())


def get_all_cases() -> list:
    """Return all test cases as flat list with task metadata."""
    all_cases = []
    for task_name, task in TASKS.items():
        for case in task["cases"]:
            all_cases.append({
                "task": task_name,
                "metric": task["metric"],
                **case,
            })
    return all_cases
