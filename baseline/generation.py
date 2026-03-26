from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from baseline.common import best_torch_dtype, detect_device, entropy_from_logits


@dataclass
class GenerationResult:
    text: str
    token_texts: List[str]
    token_logprobs: List[float]
    token_entropies: List[float]


class HFGenerator:
    def __init__(self, model_id: str, device: str = "auto") -> None:
        self.model_id = model_id
        self.device = detect_device(device)
        self.dtype = best_torch_dtype(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.is_encoder_decoder = getattr(config, "is_encoder_decoder", False)
        model_cls = AutoModelForSeq2SeqLM if self.is_encoder_decoder else AutoModelForCausalLM

        if self.device == "cuda":
            self.model = model_cls.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            self.model = model_cls.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=self.dtype,
            )
            self.model.to(self.device)
        self.model.eval()

        self._has_chat_template = (
            hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None
        )
        self._max_model_len = getattr(config, "max_position_embeddings", 8192)

    def _apply_chat_template(self, prompt: str) -> str:
        if not self._has_chat_template:
            return prompt
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        # Qwen3 defaults to thinking mode which adds <think>...</think> tags.
        # Disable it for deterministic baseline outputs.
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 96,
        return_scores: bool = False,
        stop_strings: Optional[Sequence[str]] = None,
    ) -> GenerationResult:
        formatted = self._apply_chat_template(prompt)
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_model_len - max_new_tokens,
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=return_scores,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        sequences = outputs.sequences
        generated_ids = sequences[:, prompt_len:] if not self.is_encoder_decoder else sequences
        token_ids = generated_ids[0].tolist()
        token_texts = [self.tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        token_logprobs: List[float] = []
        token_entropies: List[float] = []
        if return_scores and outputs.scores:
            transition_scores = self.model.compute_transition_scores(sequences, outputs.scores, normalize_logits=True)
            token_logprobs = [float(score) for score in transition_scores[0][-len(token_ids):]]
            token_entropies = [entropy_from_logits(logits[0]) for logits in outputs.scores]

        if stop_strings:
            for stop_string in stop_strings:
                if stop_string in text:
                    text = text.split(stop_string, 1)[0]
                    break

        return GenerationResult(
            text=text.strip(),
            token_texts=token_texts,
            token_logprobs=token_logprobs,
            token_entropies=token_entropies,
        )
