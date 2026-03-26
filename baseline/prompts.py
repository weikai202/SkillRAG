from __future__ import annotations


def format_passages(passages: list[str]) -> str:
    return "\n".join(f"passage {idx + 1}: {text}" for idx, text in enumerate(passages))


def direct_answer_prompt(question: str) -> str:
    return f"""You are answering a question-answering benchmark.
Think briefly, then give the final answer in 5 words or fewer.

Question: {question}
Rationale:
Answer:"""


def retrieval_answer_prompt(question: str, passages: list[str]) -> str:
    return f"""You are answering a question-answering benchmark using retrieved passages.
Use only the evidence that is helpful. Then give the final answer in 5 words or fewer.

Question: {question}
Passages:
{format_passages(passages)}
Rationale:
Answer:"""


def adaptive_router_prompt(question: str) -> str:
    return f"""Classify the question into exactly one label:
- NO_RETRIEVAL: the answer is likely known directly by the model
- SINGLE_HOP: one retrieval call should be enough
- MULTI_HOP: multiple retrieval steps are likely needed

Question: {question}
Label:"""


def adaptive_followup_query_prompt(question: str, draft_answer: str, passages: list[str]) -> str:
    return f"""Write one short search query that would help answer the question better.

Question: {question}
Current draft answer:
{draft_answer}

Retrieved passages:
{format_passages(passages)}

Search Query:"""


def flare_query_reformulation_prompt(question: str, masked_text: str) -> str:
    return f"""User input: Generate a summary about Joe Biden
Generated output so far: Joe Biden attended ____, where he earned a law degree.
Given the above passage, ask a question to which the answer is the term/entity/phrase: What university did Joe Biden attend?

User input: {question}
Generated output so far: {masked_text}
Given the above passage, ask a question to which the answer is the term/entity/phrase:"""


def flare_ground_sentence_prompt(low_confidence_sentence: str, passages: list[str]) -> str:
    return f"""low confidence sentence: Joe Biden attended the University of Pennsylvania, where he earned a law degree
passage 1: Joe Biden attended the University of Delaware, where he graduated in 1965 with a Bachelor of Arts in history and political science.
passage 2: After completing his undergraduate degree, Biden attended Syracuse University College of Law, where he earned a law degree in 1968.
passage 3: Joe Biden began his political career shortly after law school, becoming one of the youngest senators in U.S. history when he was elected to the Senate in 1972.
passage 4: Throughout his long political career, Biden served as the Vice President of the United States from 2009 to 2017 under President Barack Obama, and later became the 46th President of the United States in 2021.
passage 5: Biden's time at Syracuse University was marked by a plagiarism controversy, but he eventually graduated and went on to start his political career.
new sentence: He graduated from the University of Delaware in 1965 with a Bachelor of Arts in history and political science.

low confidence sentence: {low_confidence_sentence}
{format_passages(passages)}
new sentence:"""


def dragin_query_prompt(question: str, partial_answer: str, uncertain_span: str) -> str:
    return f"""Write one short BM25-friendly search query to verify or complete the uncertain part of a draft answer.

Question: {question}
Draft answer:
{partial_answer}

Uncertain span:
{uncertain_span}

Search Query:"""
