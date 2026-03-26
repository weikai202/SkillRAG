from __future__ import annotations

from typing import List

from baseline.common import BaselineTrace, clean_search_query, extract_answer_text, safe_mean, split_sentences
from baseline.generation import HFGenerator
from baseline.indexing import retrieve_texts
from baseline.prompts import (
    adaptive_followup_query_prompt,
    adaptive_router_prompt,
    direct_answer_prompt,
    dragin_query_prompt,
    flare_ground_sentence_prompt,
    flare_query_reformulation_prompt,
    retrieval_answer_prompt,
)


def _sentence_from_generation(result_text: str) -> str:
    sentences = split_sentences(result_text)
    if sentences:
        return sentences[0]
    return result_text.strip()


def _mask_low_confidence_tokens(token_texts: List[str], token_scores: List[float], keep_ratio: float = 0.5) -> str:
    if not token_texts:
        return ""
    if not token_scores:
        return "".join(token_texts).strip()
    threshold_index = max(1, int(len(token_scores) * keep_ratio))
    sorted_scores = sorted(token_scores)
    threshold = sorted_scores[min(threshold_index - 1, len(sorted_scores) - 1)]
    masked = []
    for token, score in zip(token_texts, token_scores):
        masked.append(token if score >= threshold else " __ ")
    return "".join(masked).strip()


def _parse_adaptive_label(raw_text: str) -> str | None:
    upper = raw_text.upper()
    labels = ["NO_RETRIEVAL", "SINGLE_HOP", "MULTI_HOP"]
    matches = [label for label in labels if label in upper]
    if len(matches) == 1:
        return matches[0]
    return None


class BaselineRunner:
    def __init__(
        self,
        generator: HFGenerator,
        retriever,
        top_k: int = 5,
        max_rounds: int = 3,
        flare_confidence_threshold: float = -2.5,
        dragin_entropy_threshold: float = 2.5,
    ) -> None:
        self.generator = generator
        self.retriever = retriever
        self.top_k = top_k
        self.max_rounds = max_rounds
        self.flare_confidence_threshold = flare_confidence_threshold
        self.dragin_entropy_threshold = dragin_entropy_threshold

    def retrieve(self, query: str) -> List[str]:
        return retrieve_texts(self.retriever, query)[: self.top_k]

    def direct_answer(self, question: str, max_new_tokens: int = 96) -> BaselineTrace:
        result = self.generator.generate(direct_answer_prompt(question), max_new_tokens=max_new_tokens)
        answer = extract_answer_text(result.text)
        return BaselineTrace(method="direct", raw_output=result.text, answer=answer, retrieval_count=0)

    def answer_with_retrieval(self, question: str, query: str) -> BaselineTrace:
        passages = self.retrieve(query)
        result = self.generator.generate(retrieval_answer_prompt(question, passages), max_new_tokens=128)
        answer = extract_answer_text(result.text)
        return BaselineTrace(
            method="retrieval",
            raw_output=result.text,
            answer=answer,
            retrieval_count=1,
            queries=[query],
            passages=[passages],
        )

    def run_adaptive(self, question: str) -> BaselineTrace:
        router = self.generator.generate(adaptive_router_prompt(question), max_new_tokens=12)
        route_label = _parse_adaptive_label(router.text)

        if route_label == "NO_RETRIEVAL":
            trace = self.direct_answer(question)
            trace.method = "adaptive"
            trace.steps.append({"stage": "router", "label": route_label, "raw": router.text})
            return trace
        if route_label == "SINGLE_HOP":
            trace = self.answer_with_retrieval(question, question)
            trace.method = "adaptive"
            trace.steps.append({"stage": "router", "label": route_label, "raw": router.text})
            return trace
        if route_label is None:
            trace = self.answer_with_retrieval(question, question)
            trace.method = "adaptive"
            trace.steps.append(
                {
                    "stage": "router",
                    "label": "INVALID_ROUTE",
                    "raw": router.text,
                    "fallback": "SINGLE_HOP",
                }
            )
            return trace

        trace = BaselineTrace(method="adaptive", raw_output="", answer="", retrieval_count=0)
        trace.steps.append({"stage": "router", "label": route_label, "raw": router.text})
        current_query = question
        last_output = ""
        for round_id in range(1, self.max_rounds + 1):
            passages = self.retrieve(current_query)
            output = self.generator.generate(retrieval_answer_prompt(question, passages), max_new_tokens=128)
            answer = extract_answer_text(output.text)
            trace.retrieval_count += 1
            trace.queries.append(current_query)
            trace.passages.append(passages)
            trace.steps.append(
                {
                    "stage": f"round_{round_id}",
                    "query": current_query,
                    "raw_output": output.text,
                    "answer": answer,
                }
            )
            last_output = output.text
            trace.answer = answer
            trace.raw_output = output.text
            if round_id == self.max_rounds:
                break
            next_query_output = self.generator.generate(
                adaptive_followup_query_prompt(question, output.text, passages),
                max_new_tokens=32,
            )
            next_query = clean_search_query(next_query_output.text, fallback=question)
            if next_query == current_query:
                break
            current_query = next_query
        if not trace.raw_output:
            trace.raw_output = last_output
        return trace

    def run_dragin(self, question: str) -> BaselineTrace:
        draft = self.generator.generate(direct_answer_prompt(question), max_new_tokens=128, return_scores=True)
        answer = extract_answer_text(draft.text)
        trace = BaselineTrace(method="dragin", raw_output=draft.text, answer=answer, retrieval_count=0)
        draft_sentence = _sentence_from_generation(draft.text)
        entropy_signal = safe_mean(draft.token_entropies[: max(1, min(24, len(draft.token_entropies)))])
        trace.steps.append(
            {
                "stage": "draft",
                "raw_output": draft.text,
                "draft_sentence": draft_sentence,
                "entropy_signal": entropy_signal,
            }
        )

        if entropy_signal <= self.dragin_entropy_threshold:
            return trace

        current_query = question
        if draft_sentence:
            query_output = self.generator.generate(
                dragin_query_prompt(question, draft.text, draft_sentence),
                max_new_tokens=24,
            )
            current_query = clean_search_query(query_output.text, fallback=question)

        for round_id in range(1, self.max_rounds + 1):
            passages = self.retrieve(current_query)
            grounded = self.generator.generate(retrieval_answer_prompt(question, passages), max_new_tokens=128, return_scores=True)
            answer = extract_answer_text(grounded.text)
            trace.retrieval_count += 1
            trace.queries.append(current_query)
            trace.passages.append(passages)
            trace.raw_output = grounded.text
            trace.answer = answer
            current_entropy = safe_mean(grounded.token_entropies[: max(1, min(24, len(grounded.token_entropies)))])
            trace.steps.append(
                {
                    "stage": f"round_{round_id}",
                    "query": current_query,
                    "raw_output": grounded.text,
                    "entropy_signal": current_entropy,
                }
            )
            if current_entropy <= self.dragin_entropy_threshold:
                break
            next_query_out = self.generator.generate(
                dragin_query_prompt(question, grounded.text, _sentence_from_generation(grounded.text)),
                max_new_tokens=24,
            )
            next_query = clean_search_query(next_query_out.text, fallback=current_query)
            if next_query == current_query:
                break
            current_query = next_query
        return trace

    def run_flare(self, question: str) -> BaselineTrace:
        draft = self.generator.generate(direct_answer_prompt(question), max_new_tokens=96, return_scores=True)
        answer = extract_answer_text(draft.text)
        trace = BaselineTrace(method="flare", raw_output=draft.text, answer=answer, retrieval_count=0)

        prefix_tokens = draft.token_texts[: max(1, min(32, len(draft.token_texts)))]
        prefix_scores = draft.token_logprobs[: len(prefix_tokens)]
        confidence = safe_mean(prefix_scores) if prefix_scores else 0.0
        trace.steps.append({"stage": "draft", "raw_output": draft.text, "confidence": confidence})
        if confidence >= self.flare_confidence_threshold:
            return trace

        masked_text = _mask_low_confidence_tokens(prefix_tokens, prefix_scores or [0.0] * len(prefix_tokens))
        query_generation = self.generator.generate(
            flare_query_reformulation_prompt(question, masked_text),
            max_new_tokens=32,
        )
        retrieval_query = clean_search_query(query_generation.text, fallback=question)

        passages = self.retrieve(retrieval_query)
        trace.retrieval_count += 1
        trace.queries.append(retrieval_query)
        trace.passages.append(passages)

        low_conf_sentence = _sentence_from_generation(draft.text)
        grounded_sentence = self.generator.generate(
            flare_ground_sentence_prompt(low_conf_sentence, passages),
            max_new_tokens=48,
        )
        final = self.generator.generate(retrieval_answer_prompt(question, passages), max_new_tokens=128)
        trace.raw_output = final.text
        trace.answer = extract_answer_text(final.text)
        trace.steps.append(
            {
                "stage": "flare_rewrite",
                "masked_text": masked_text,
                "retrieval_query": retrieval_query,
                "grounded_sentence": grounded_sentence.text,
                "final_output": final.text,
            }
        )
        return trace

    def run(self, method: str, question: str) -> BaselineTrace:
        if method == "adaptive":
            return self.run_adaptive(question)
        if method == "dragin":
            return self.run_dragin(question)
        if method == "flare":
            return self.run_flare(question)
        raise ValueError(f"Unsupported method: {method}")
