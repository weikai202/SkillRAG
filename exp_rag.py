#%%
import argparse
from argparse import Namespace

import time
import json
import ast
import os
import numpy as np
from tqdm import tqdm
from functools import partial

import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM, PreTrainedTokenizerBase
from transformers import StoppingCriteria, StoppingCriteriaList

from transformer_lens import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
import transformer_lens.utils as utils
from transformer_lens.utilities import devices

import torch
from torch.utils.data import DataLoader, Dataset

from metrics.metrics import EmF1Metric, SupportEmF1Metric

from utils import AttnWeightRAG, FixLengthRAG, StopOnPunctuationWithLogit, Config_Maker, preprocessing, batch_topk_sim
from utils import load_prober_cfg_for_model, load_prober_models, return_prober_logit_gemma_2b, evaluator, normalize_answer
from prompts import inst_prompt, cot_prompt, retr_qa, retr_qa_cot2
from prompts import skillrag_diagnosis_prompt, skillrag_router_prompt
from prompts import skillrag_query_rewrite_prompt, skillrag_decomposition_prompt, skillrag_evidence_grounded_prompt, skillrag_insufficient_evidence_prompt

from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast, overload
from typing_extensions import Literal
from jaxtyping import Float, Int

from transformer_lens.utils import USE_DEFAULT_VALUE
SUPPORTED_PROBER_MODELS = ['google/gemma-2b', 'meta-llama/Meta-Llama-3-8B-Instruct', 'Qwen/Qwen3-8B', 'google/gemma-2-9b-it']

class CustomHookedTransformer(HookedTransformer):
    def __init__(
        self,
        cfg: Union[HookedTransformerConfig, Dict],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        move_to_device: bool = True,
        default_padding_side: Literal["left", "right"] = "right",
    ):
        super().__init__(cfg, tokenizer, move_to_device, default_padding_side)

    @torch.inference_mode()
    def generate(
        self,
        input: Union[str, Float[torch.Tensor, "batch pos"]] = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        use_past_kv_cache: bool = True,
        prepend_bos: Optional[bool] = USE_DEFAULT_VALUE,
        padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
        return_type: Optional[str] = "input",
        verbose: bool = True,
        stop_tokens: Optional[List[str]] = None,
        stop_tokenss: Optional[List[str]] = None,
    ) -> Union[Int[torch.Tensor, "batch pos_plus_new_tokens"], str]:

        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            if type(input) == str:
                # If text, convert to tokens (batch_size=1)
                assert (
                    self.tokenizer is not None
                ), "Must provide a tokenizer if passing a string to the model"
                tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
            else:
                tokens = input

            if return_type == "input":
                if type(input) == str:
                    return_type = "str"
                else:
                    return_type = "tensor"

            assert isinstance(tokens, torch.Tensor)
            batch_size, ctx_length = tokens.shape
            device = devices.get_device_for_block_index(0, self.cfg)
            tokens = tokens.to(device)
            if use_past_kv_cache:
                past_kv_cache = HookedTransformerKeyValueCache.init_cache(
                    self.cfg, self.cfg.device, batch_size
                )
            else:
                past_kv_cache = None

            stop_tokens: List[int] = []
            eos_token_for_padding = 0
            assert self.tokenizer is not None
            if stop_at_eos:
                tokenizer_has_eos_token = (
                    self.tokenizer is not None and self.tokenizer.eos_token_id is not None
                )
                if eos_token_id is None:
                    assert (
                        tokenizer_has_eos_token
                    ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"

                    eos_token_id = self.tokenizer.eos_token_id

                if isinstance(eos_token_id, int):
                    stop_tokens = [eos_token_id]
                    eos_token_for_padding = eos_token_id
                else:
                    # eos_token_id is a Sequence (e.g. list or tuple)
                    stop_tokens = eos_token_id
                    eos_token_for_padding = (
                        self.tokenizer.eos_token_id if tokenizer_has_eos_token else eos_token_id[0]
                    )

            # An array to track which sequences in the batch have finished.
            finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.cfg.device)

            # Currently nothing in HookedTransformer changes with eval, but this is here in case
            # that changes in the future.
            self.eval()
            count = 0
            for index in tqdm(range(max_new_tokens), disable=not verbose):
                # While generating, we keep generating logits, throw away all but the final logits,
                # and then use those logits to sample from the distribution We keep adding the
                # sampled tokens to the end of tokens.
                if use_past_kv_cache:
                    # We just take the final tokens, as a [batch, 1] tensor
                    if index > 0:
                        logits = self.forward(
                            tokens[:, -1:],
                            return_type="logits",
                            prepend_bos=prepend_bos,
                            padding_side=padding_side,
                            past_kv_cache=past_kv_cache,
                        )
                    else:
                        logits = self.forward(
                            tokens,
                            return_type="logits",
                            prepend_bos=prepend_bos,
                            padding_side=padding_side,
                            past_kv_cache=past_kv_cache,
                        )
                else:
                    # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
                    # the cache.
                    logits = self.forward(
                        tokens,
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                    )
                final_logits = logits[:, -1, :]

                if do_sample:
                    sampled_tokens = utils.sample_logits(
                        final_logits,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        freq_penalty=freq_penalty,
                        tokens=tokens,
                    ).to(devices.get_device_for_block_index(0, self.cfg))
                else:
                    sampled_tokens = final_logits.argmax(-1).to(
                        devices.get_device_for_block_index(0, self.cfg)
                    )

                if stop_at_eos:
                    # For all unfinished sequences, add on the next token. If a sequence was
                    # finished, throw away the generated token and add eos_token_for_padding
                    # instead.
                    sampled_tokens[finished_sequences] = eos_token_for_padding
                    finished_sequences.logical_or_(
                        torch.isin(
                            sampled_tokens.to(self.cfg.device),
                            torch.tensor(stop_tokens).to(self.cfg.device),
                        )
                    )

                tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)
                
                if stop_tokenss:
                    generated_text = self.tokenizer.decode(tokens[0])
                    if 5 != len(generated_text.split('\n\n')):
                        break
                    # if any(stop_token in generated_text for stop_token in stop_tokenss):
                    #     import pdb;pdb.set_trace()
                    #     count +=1
                    #     if count ==6:
                    #         break
                if stop_at_eos and finished_sequences.all():
                    break
                
            if return_type == "str":
                generated_text = self.tokenizer.decode(tokens[0, 1:] if self.cfg.default_prepend_bos else tokens[0])
                if stop_tokenss:
                    for stop_token in stop_tokenss:
                        if stop_token in generated_text:
                            generated_text = generated_text.split(stop_token)[0]
                            break
                return generated_text
            else:
                return tokens
#%%
def main(args):
    steps_limit =args.steps_limit # 100
    threshold = args.threshold # 0.5
    is_sparse = args.is_sparse # True
    retr_method = args.retr_method  # probing, none, simple
    position = args.position # 'resid_post'
    dataset_name = args.dataset_name
    is_cot = args.is_cot
    model_id = args.model_id
    tr_or_dev = args.tr_or_dev
    _ds = args.ds # 25, 50, 75, 1000, else
    metric = EmF1Metric()
    print('*'*70)
    print(f"threshold: {threshold}, retr_method: {retr_method}, position: {position},\ndataset_name: {dataset_name}, model_id: {model_id}, steps_limit: {steps_limit} \n ablation: {args.ablation}, prober_ds_len: {_ds}")
    print('*'*70)
    #%%
    if is_cot:
        prompt_function_data=cot_prompt
        prompt_function_retr = retr_qa_cot2
        savename_is_cot = 'cot'
        max_new_tokens = 150
        
    if is_sparse:
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core import Document
        retr_type = 'sparse'
        print('sparse retrieval loading...')
        docstore2 = SimpleDocumentStore.from_persist_path(f"raw_data/sparse_index/llama_index_bm25_model_{dataset_name}_2.json") #
        bm25=BM25Retriever.from_defaults(docstore=docstore2, similarity_top_k=5)
    else:
        retr_type = 'dense'
        print('dense retrieval loading...')
        model_retr_id = 'facebook/contriever-msmarco'
        model_retr = SentenceTransformer(model_retr_id)
        index = faiss.read_index(f'raw_data/dense_index/contriever_{dataset_name}_2.bin') #
    print('finish!!')
    print('*'*70)
    if (dataset_name =='hotpotqa') and (tr_or_dev=='dev'): path = f'raw_data/hotpotqa/hotpot_{tr_or_dev}_distractor_v1.json'
    elif (dataset_name =='hotpotqa') and (tr_or_dev=='train'): path = f'raw_data/hotpotqa/hotpot_{tr_or_dev}_v1.1.json'
    elif dataset_name =='nq': path = f'raw_data/nq/biencoder-nq-{tr_or_dev}.json'
    elif dataset_name =='trivia': path = f'raw_data/trivia/biencoder-trivia-{tr_or_dev}.json'
    elif dataset_name =='2wikimultihopqa': path = f'raw_data/2wikimultihopqa/{tr_or_dev}.json' # TODO einsum error fix when do model.generate
    elif dataset_name =='musique': path = f'raw_data/musique/musique_full_v1.0_{tr_or_dev}.jsonl'
    elif dataset_name == 'iirc': path = f"raw_data/iirc/{tr_or_dev}.json"
    
    if (args.dataset_name == 'hotpotqa') or (args.dataset_name == '2wikimultihopqa') or (args.dataset_name == 'musique') or (args.dataset_name == 'iirc'):
        metric = SupportEmF1Metric()    
        answer_name = 'answer'
    else:
        metric = EmF1Metric()
        answer_name = 'answers'
    
    dataset = []
    if dataset_name =='musique':
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
    else:
        with open(path, 'r', encoding='utf-8-sig') as f:  #문제
            js = json.load(f) 
        if dataset_name == 'iirc':
            for tmp in tqdm(js):
                for example in tmp['questions']:
                    qid = example["qid"]
                    question = example['question']

                    ans = example['answer']

                    if ans['type'] == 'none':
                        continue
                    elif ans['type'] == 'value' or ans['type'] == 'binary':
                        answer = [ans['answer_value']]
                    elif ans['type'] == 'span':
                        answer = [v['text'].strip() for v in ans['answer_spans']]
                    
                    # context = example['context']
                    dataset.append({
                        'qid': qid,
                        'question': question,
                        'answer': answer,
                        # 'ctxs': context,
                    })                
        else: dataset = js
    if is_sparse: pass
    else: corpus = pd.read_csv(f'raw_data/{dataset_name}_index_2.csv') 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_id == 'Qwen/Qwen3-8B':
        # transformer_lens expects hf_config.rope_theta for rotary_base conversion.
        try:
            from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
            if not hasattr(Qwen3Config, 'rope_theta'):
                Qwen3Config.rope_theta = 1000000
        except Exception:
            pass

    model_load_kwargs = {
        'device': device,
        'trust_remote_code': True,
    }
    config_kwargs = {'trust_remote_code': True}
    if model_id == 'Qwen/Qwen3-8B':
        # Qwen3Config in some transformer_lens versions misses rope_theta expectation.
        config_kwargs['rope_theta'] = 1000000
    try:
        model = CustomHookedTransformer.from_pretrained(
            model_id,
            config_kwargs=config_kwargs,
            **model_load_kwargs,
        )
    except TypeError:
        # Fallback for transformer_lens versions without config_kwargs support.
        model = CustomHookedTransformer.from_pretrained(model_id, **model_load_kwargs)
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    if retr_method == 'probing':


        if model_id in SUPPORTED_PROBER_MODELS:
            cfg_list = load_prober_cfg_for_model(model, Config_Maker, position, device)
            probers = load_prober_models(_ds, cfg_list)
            layer_configs = cfg_list
            
        cache = {}

        def hook_fn(activations, hook, layer):
            if layer not in cache:
                cache[layer] = []
            cache[layer].append(activations.detach().cpu())
            return activations

        def add_layer_hook(model, layer_name):
            hook = partial(hook_fn, layer=layer_name)
            model.add_hook(layer_name, hook)

        for prober_cfg in layer_configs:
            layer_name = f'blocks.{prober_cfg.layer}.hook_{prober_cfg.position}'
            add_layer_hook(model, layer_name)
    if retr_method == 'skillrag':
        if model_id in SUPPORTED_PROBER_MODELS:
            cfg_list = load_prober_cfg_for_model(model, Config_Maker, position, device)
            probers = load_prober_models(_ds, cfg_list)
            layer_configs = cfg_list
        cache = {}

        def hook_fn(activations, hook, layer):
            if layer not in cache:
                cache[layer] = []
            cache[layer].append(activations.detach().cpu())
            return activations

        def add_layer_hook(model, layer_name):
            hook = partial(hook_fn, layer=layer_name)
            model.add_hook(layer_name, hook)

        for prober_cfg in layer_configs:
            layer_name = f'blocks.{prober_cfg.layer}.hook_{prober_cfg.position}'
            add_layer_hook(model, layer_name)
        
    model.eval()
        
    if args.extract_sep:
        dataset = dataset[args.sep_number:]
        save_data_name = f'after{args.sep_number}'
        
    questions, answers = [], []
    for value in tqdm(dataset):
        question, answer = value['question'], value[f'{answer_name}']
        questions.append(question)
        answers.append(answer)
    df = pd.DataFrame([questions, answers]).T
    df.columns = ['query', 'answer']

    class CustomDataset(Dataset):
        def __init__(self, dataset, prompt, tokenizer):
            self.dataset = dataset
            self.prompt = prompt
            self.tokenizer = tokenizer
        def __len__(self):
            return len(self.dataset)
                    
        def __getitem__(self, index):
            item = self.dataset[index]
            prompt_text = self.prompt(item)
            token = self.tokenizer(prompt_text,return_tensors='pt').to(device)
            return {
                'input_ids': token['input_ids'].squeeze(),
                'attention_mask': token['attention_mask'].squeeze(),
                'prompt_text': prompt_text,
                'text': item,
            }
    df = preprocessing(df, args)
    if (retr_method == 'flare') or (retr_method == 'linguistic'):
        pass
    else:
        dataloader=DataLoader(CustomDataset(df['query'], prompt = prompt_function_data, tokenizer=tokenizer))

    def return_evidences(retrieved_passages, is_sparse = args.is_sparse):
        evidences = ''
        def return_evidence(evidence, is_sparse):
            if is_sparse: return evidence.text
            else: return evidence
        for num, evidence in enumerate(retrieved_passages):
            if (num+1) == len(retrieved_passages):
                evidences+= f'passage {num+1}: {return_evidence(evidence,is_sparse)}'
            else:
                evidences+= f'passage {num+1}: {return_evidence(evidence,is_sparse)}'+'\n'
        return evidences
    
    def return_mean_output(prober_cfg, prober):
        layer_name = f'blocks.{prober_cfg.layer}.hook_{prober_cfg.position}'
        with torch.no_grad():
            # import pdb;pdb.set_trace()
            input=torch.concat(cache[layer_name][1:], dim = 1).to(device)
            input = torch.sum(input, dim = 1)
            logit=prober(input)
        # import pdb;pdb.set_trace()
        return logit
    
    def extract_pred_for_eval(pred):
        if args.is_cot:
            try:
                pred = pred.split('\n\n')[4]
            except:
                pred = pred
            if len(pred.split('\n')) > 7:
                new_pred = '\n'.join(pred.split('\n')[8:])
            else:
                new_pred = '\n'.join(pred.split('\n')[1:])
            return new_pred.replace('</s>','').replace('<eos>','').replace('Answer:','').strip()
        else:
            try:
                new_pred = pred.split('\n\n')[2]
            except:
                new_pred = pred
            return new_pred.replace('</s>','').replace('<eos>','').replace('Answer:','').strip()
    
    def prober_need_retrieval():
        if model_id in SUPPORTED_PROBER_MODELS:
            logits = return_prober_logit_gemma_2b(return_mean_output, cfg_list, probers)
            for_set_threshold = torch.zeros_like(logits[0].squeeze())
            for num in range(args.ablation, len(logits)):
                for_set_threshold += (softmax_f(logits[num])).squeeze()
        else:
            assert 'model id error...'

        if for_set_threshold[0].item() + threshold < for_set_threshold[1].item():
            return 0, for_set_threshold
        return 1, for_set_threshold

    def diagnose_failure(question, reasoning_answer):
        def _tokenize_with_ctx_limit(prompt_text, reserve_new_tokens):
            max_ctx = getattr(model.cfg, "n_ctx", 2048)
            max_in = max(32, int(max_ctx) - int(reserve_new_tokens))
            return tokenizer(
                prompt_text,
                return_tensors='pt',
                truncation=True,
                max_length=max_in,
            )['input_ids'].to(device)

        diagnosis_prompt = skillrag_diagnosis_prompt(question, reasoning_answer)
        diagnosis_input_ids = _tokenize_with_ctx_limit(diagnosis_prompt, reserve_new_tokens=48)
        with torch.no_grad():
            diagnosis_out = model.generate(diagnosis_input_ids, do_sample=False, max_new_tokens=48)
        diagnosis_text = tokenizer.decode(diagnosis_out[0][diagnosis_input_ids.shape[1]:], skip_special_tokens=True)
        return diagnosis_text.replace('</s>','').replace('<eos>','').strip()

    def choose_skill(question, reasoning_answer):
        diagnosis = diagnose_failure(question, reasoning_answer)
        router_prompt = skillrag_router_prompt(question, reasoning_answer, diagnosis)
        max_ctx = getattr(model.cfg, "n_ctx", 2048)
        max_in = max(32, int(max_ctx) - 20)
        router_input_ids = tokenizer(
            router_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=max_in,
        )['input_ids'].to(device)
        with torch.no_grad():
            route_out = model.generate(router_input_ids, do_sample=False, max_new_tokens=20)
        route_text = tokenizer.decode(route_out[0][router_input_ids.shape[1]:], skip_special_tokens=True).lower()
        if 'insufficient_evidence' in route_text:
            return 'insufficient_evidence', diagnosis
        if 'multi_hop_missing' in route_text:
            return 'multi_hop_missing', diagnosis
        if 'evidence_not_used' in route_text:
            return 'evidence_not_used', diagnosis
        return 'query_misaligned', diagnosis

    def parse_search_query(skill_name, raw_text, fallback_query):
        text = raw_text.replace('</s>', '').replace('<eos>', '').strip()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) == 0:
            return fallback_query

        for key in ['Final Search Query:', 'Search Query:', 'Query:']:
            for line in lines:
                if key.lower() in line.lower():
                    return line.split(':', 1)[1].strip() if ':' in line else line

        if skill_name == 'multi_hop_missing':
            sub_queries = []
            for line in lines:
                low = line.lower()
                if low.startswith('sub-query') or line.endswith('?') or line.startswith('-'):
                    sub_queries.append(line.split(':', 1)[1].strip() if ':' in line else line.lstrip('-').strip())
            if len(sub_queries) > 0:
                return ' '.join(sub_queries[:3]).strip()

        return lines[0]

    def skill_generate_search_query(skill_name, question, reasoning_answer, evidences):
        if skill_name == 'query_misaligned':
            generation_prompt = skillrag_query_rewrite_prompt(question, reasoning_answer, evidences)
        elif skill_name == 'multi_hop_missing':
            generation_prompt = skillrag_decomposition_prompt(question, reasoning_answer, evidences)
        elif skill_name == 'insufficient_evidence':
            generation_prompt = skillrag_insufficient_evidence_prompt(question, reasoning_answer, evidences)
        else:
            generation_prompt = skillrag_evidence_grounded_prompt(question, reasoning_answer, evidences)
        max_ctx = getattr(model.cfg, "n_ctx", 2048)
        max_in = max(32, int(max_ctx) - 96)
        generation_input_ids = tokenizer(
            generation_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=max_in,
        )['input_ids'].to(device)
        with torch.no_grad():
            generated = model.generate(generation_input_ids, do_sample=False, max_new_tokens=96)
        generated_text = tokenizer.decode(generated[0][generation_input_ids.shape[1]:], skip_special_tokens=True)
        return parse_search_query(skill_name, generated_text, question), generated_text

    retr_count_list, pred_list = [], []
    skillrag_initial_outputs = []
    skillrag_round_logs = []
    steps = 0
    softmax_f = torch.nn.Softmax(dim = 1)
    if retr_method == 'probing':
        start = time.time()
        for value in tqdm(dataloader):
            cache={}
            
            retr_count = 0
            with torch.no_grad():
                output = model.generate(value['input_ids'], do_sample=False, max_new_tokens=max_new_tokens, stop_tokenss=["Question:"])
                if steps % 10 == 0:
                    print(model.to_string(output)[0])
                
            if model_id in SUPPORTED_PROBER_MODELS:
                logits = return_prober_logit_gemma_2b(return_mean_output, cfg_list, probers)
                for_set_threshold = torch.zeros_like(logits[0].squeeze())
                
                for num in range(args.ablation, len(logits)):
                    for_set_threshold += (softmax_f(logits[num])).squeeze() 
                
            else: assert 'model id error...'
            
            if for_set_threshold[0].item() + threshold < for_set_threshold[1].item(): prediction_do_more_retriever = 0 # + args.threshold
            else: prediction_do_more_retriever=1
            
            if prediction_do_more_retriever == 0:
                # print(model.to_string(output))
                pred_list.append(model.to_string(output)[0])
                print(for_set_threshold[0].item() + threshold,for_set_threshold[1].item())
            else:
                while prediction_do_more_retriever == 1:
                    cache={}
                    if is_sparse:
                        if retr_count == 0:
                            retrieved_passages = bm25.retrieve(value['text'][0])
                        else:
                            retrieved_passages = bm25.retrieve(search_input_new[0])
                        evidences = return_evidences(retrieved_passages)
                    else:
                        if retr_count == 0:
                            D, I = batch_topk_sim(model_retr, value['text'], index, k = 5)
                        else:
                            
                            D, I = batch_topk_sim(model_retr, search_input_new, index, k = 5)
                        retrieved_passages = list(corpus.iloc[I[0].tolist(),0])
                        
                        evidences = return_evidences(retrieved_passages)
                    new_input = prompt_function_retr(value['text'][0], evidences)
                    
                    with torch.no_grad():
                        
                        output = model.generate(tokenizer(new_input, return_tensors='pt')['input_ids'].to(device), do_sample=False, max_new_tokens=max_new_tokens, stop_tokenss=["Question:"])
                        output.to('cpu')
                        
                        if model_id in SUPPORTED_PROBER_MODELS:
                            logits = return_prober_logit_gemma_2b(return_mean_output, cfg_list, probers)
                            for_set_threshold = torch.zeros_like(logits[0].squeeze())
                            for num in range(args.ablation, len(logits)):
                                for_set_threshold += (softmax_f(logits[num])).squeeze() 
                            
                        else: assert 'model id error...'
                        
                        if for_set_threshold[0].item() + threshold < for_set_threshold[1].item(): prediction_do_more_retriever = 0 #  + args.threshold
                        else: prediction_do_more_retriever=1
                        
                        search_input_new=model.to_string(output)
                        if (steps + 1) % 3 == 0:
                            print(search_input_new[0])
                        print(for_set_threshold[0].item() + threshold,for_set_threshold[1].item())
                        
                        if retr_count > 2:
                            break
                        else:
                            retr_count += 1
                pred_list.append(search_input_new[0])
                
            retr_count_list.append(retr_count)
            steps += 1
            print(steps)
                
            if steps > steps_limit:   
                end = time.time()
                break

    if retr_method == 'none':
        start = time.time()
        for value in tqdm(dataloader):
            
            with torch.no_grad():
                output = model.generate(value['input_ids'], do_sample=False, max_new_tokens=max_new_tokens, stop_tokenss=["Question:"])
            pred_list.append(model.to_string(output)[0])
            steps +=1
            if steps > steps_limit:   
                end = time.time()
                break
            
    if retr_method =='simple':
        start = time.time()
        for value in tqdm(dataloader):
            if is_sparse:
                retrieved_passages = bm25.retrieve(value['text'][0])
                evidences = return_evidences(retrieved_passages)
                    
            else:
                D, I = batch_topk_sim(model_retr, value['text'], index, k = 5)
                retrieved_passages = list(corpus.iloc[I[0].tolist(),0])
                evidences = return_evidences(retrieved_passages)
                    
            new_input = prompt_function_retr(value['text'][0], evidences)

            output = model.generate(tokenizer(new_input, return_tensors='pt')['input_ids'].to(device), do_sample=False, max_new_tokens=max_new_tokens, stop_tokenss=["Question:"])
            output.to('cpu')

            search_input_new=model.to_string(output)[0]
            pred_list.append(search_input_new)
            steps += 1
            if steps > steps_limit:
                end = time.time()
                break
    
    if retr_method == 'skillrag':
        start = time.time()
        max_retrieval_rounds = args.max_retrieval_rounds
        for value in tqdm(dataloader):
            retr_count = 0
            final_output_text = ''
            sample_round_logs = []
            cache = {}
            with torch.no_grad():
                initial_output = model.generate(value['input_ids'], do_sample=False, max_new_tokens=max_new_tokens, stop_tokenss=["Question:"])
                initial_output.to('cpu')
            initial_output_text = model.to_string(initial_output)[0]
            prediction_do_more_retriever, initial_scores = prober_need_retrieval()
            initial_scores = [float(x) for x in initial_scores.tolist()]
            skillrag_initial_outputs.append(initial_output_text)

            if prediction_do_more_retriever == 0:
                final_output_text = initial_output_text
                sample_round_logs.append({
                    'stage': 'initial',
                    'prober_scores': initial_scores,
                    'prediction_do_more_retriever': int(prediction_do_more_retriever),
                    'stopped_after_initial': True,
                })
                skillrag_round_logs.append(json.dumps(sample_round_logs, ensure_ascii=False))
                pred_list.append(final_output_text)
                retr_count_list.append(retr_count)
                steps += 1
                if steps > steps_limit:
                    end = time.time()
                    break
                continue

            current_query = value['text'][0]
            need_more = 1
            last_output_text = initial_output_text
            while (need_more == 1) and (retr_count < max_retrieval_rounds):
                cache = {}
                if is_sparse:
                    retrieved_passages = bm25.retrieve(current_query)
                    evidences = return_evidences(retrieved_passages)
                else:
                    D, I = batch_topk_sim(model_retr, [current_query], index, k = 5)
                    retrieved_passages = list(corpus.iloc[I[0].tolist(),0])
                    evidences = return_evidences(retrieved_passages)

                new_input = prompt_function_retr(value['text'][0], evidences)
                with torch.no_grad():
                    output = model.generate(tokenizer(new_input, return_tensors='pt')['input_ids'].to(device), do_sample=False, max_new_tokens=max_new_tokens, stop_tokenss=["Question:"])
                    output.to('cpu')
                output_text = model.to_string(output)[0]
                last_output_text = output_text
                final_output_text = output_text
                retr_count += 1

                need_more, round_scores = prober_need_retrieval()
                round_scores = [float(x) for x in round_scores.tolist()]
                round_log = {
                    'stage': f'retrieval_round_{retr_count}',
                    'retrieval_query': current_query,
                    'evidences': evidences,
                    'output_text': output_text,
                    'prober_scores': round_scores,
                    'prediction_do_more_retriever': int(need_more),
                }

                if need_more == 0:
                    round_log['stopped_by_prober'] = True
                    sample_round_logs.append(round_log)
                    break

                selected_skill, diagnosis = choose_skill(value['text'][0], output_text)
                round_log['diagnosis'] = diagnosis
                round_log['selected_skill'] = selected_skill

                next_query, next_generation_raw = skill_generate_search_query(selected_skill, value['text'][0], output_text, evidences)
                round_log['skill_generation_raw'] = next_generation_raw
                round_log['next_retrieval_query'] = next_query
                sample_round_logs.append(round_log)
                current_query = next_query

            if final_output_text == '':
                final_output_text = last_output_text

            skillrag_round_logs.append(json.dumps(sample_round_logs, ensure_ascii=False))
            pred_list.append(final_output_text)
            retr_count_list.append(retr_count)
            steps += 1
            if steps > steps_limit:
                end = time.time()
                break
      
    
    if 'end' not in locals():
        end = time.time()
    acc, metric, pred_to_train=evaluator(df, metric, pred_list,args)

    def _row_em(pred_text, gold_answer):
        if isinstance(gold_answer, str):
            try:
                parsed = ast.literal_eval(gold_answer)
                if isinstance(parsed, list):
                    golds = [str(x) for x in parsed]
                else:
                    golds = [gold_answer]
            except Exception:
                golds = [gold_answer]
        elif isinstance(gold_answer, list):
            golds = [str(x) for x in gold_answer]
        else:
            golds = [str(gold_answer)]
        pred_norm = normalize_answer(str(pred_text))
        return int(any(pred_norm == normalize_answer(g) for g in golds))
    
    print('time: ',end-start)
    print('acc: ', sum(acc)/len(acc))
    if args.extracting_cot_qa:
        if '7' in model_id:
            _save_path = '7b'
        if '2b' in model_id:
            _save_path = '2b'
        if '8' in model_id:
            _save_path = '8b'
        if '9' in model_id:
            _save_path = '9b'
        if not args.extract_sep:
            save_data_name = 'all'

        used_len = min(len(pred_list), len(pred_to_train), len(acc), len(df))
        em_list = [
            _row_em(pred_to_train[i], df['answer'].iloc[i])
            for i in range(used_len)
        ]
        question_with_prompt = [prompt_function_data(q) for q in df['query'][:used_len]]
        dfdf = pd.DataFrame({
            'retr_method': [retr_method] * used_len,
            'question_with_prompt': question_with_prompt,
            'pred_with_prompt': pred_list[:used_len],
            'pred': pred_to_train[:used_len],
            'answer': list(df['answer'][:used_len]),
            'acc': acc[:used_len],
            'em': em_list,
        })
        if retr_method == 'skillrag':
            dfdf['initial_output'] = skillrag_initial_outputs[:used_len]
            dfdf['round_logs'] = skillrag_round_logs[:used_len]
        
        save_path = f"dataset/{_save_path}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        current_save_file = f"dataset/{_save_path}/retrieval_qa_{model_id.split('/')[1]}_{dataset_name}_{retr_method}_{tr_or_dev}_{save_data_name}_{steps_limit}.csv"
        dfdf.to_csv(current_save_file, index=False)

        if retr_method in ['simple', 'none']:
            counterpart_method = 'none' if retr_method == 'simple' else 'simple'
            counterpart_file = f"dataset/{_save_path}/retrieval_qa_{model_id.split('/')[1]}_{dataset_name}_{counterpart_method}_{tr_or_dev}_{save_data_name}_{steps_limit}.csv"
            if os.path.exists(counterpart_file):
                current_df = pd.read_csv(current_save_file).dropna(subset=['question_with_prompt', 'pred', 'acc']).reset_index(drop=True)
                other_df = pd.read_csv(counterpart_file).dropna(subset=['question_with_prompt', 'pred', 'acc']).reset_index(drop=True)
                merged_df = pd.concat([current_df, other_df], axis=0, ignore_index=True)
                acc_trace_df = (
                    merged_df
                    .pivot_table(index='question_with_prompt', columns='retr_method', values='acc', aggfunc='first')
                    .rename(columns={'none': 'none_acc', 'simple': 'simple_acc'})
                    .reset_index()
                )
                merged_df = merged_df.merge(acc_trace_df[['question_with_prompt', 'none_acc', 'simple_acc']], on='question_with_prompt', how='left')
                none_acc = merged_df['none_acc']
                simple_acc = merged_df['simple_acc']
                merged_df['need_retrieval_label'] = np.where(none_acc == 1, 0, 1)
                merged_df['unstable_case'] = np.where((none_acc == 1) & (simple_acc == 0), 1, 0)
                merged_df['use_for_training'] = np.where(merged_df['unstable_case'] == 1, 0, 1)
                merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

                if tr_or_dev == 'train':
                    merged_df.to_csv(
                        f"dataset/{_save_path}/retrieval_qa_{model_id.split('/')[1]}_{dataset_name}_all_train_in3_.csv",
                        index=False
                    )
                elif tr_or_dev == 'dev':
                    if len(merged_df) > 500:
                        test_df = merged_df.iloc[:500].reset_index(drop=True)
                    else:
                        test_df = merged_df.reset_index(drop=True)
                    test_df.to_csv(
                        f"dataset/{_save_path}/retrieval_qa_{model_id.split('/')[1]}_{dataset_name}_all_zeroshot_test_500.csv",
                        index=False
                    )
        
        print('making retrieval dataset is end !!!')
    
    else:
        if (args.dataset_name == 'hotpotqa') or (args.dataset_name == '2wikimultihopqa') or (args.dataset_name == 'musique') or (args.dataset_name == 'iirc'):
            df = pd.DataFrame([[retr_method], [end-start],[sum(acc)/len(acc)], [metric.get_metric()['title_em']], [metric.get_metric()['title_f1']]]).T
            if retr_method == 'probing':
                dfdf_clf_pred = pd.DataFrame([str(retr_count_list)])    
                dfdf_acc = pd.DataFrame([str(acc)])
                df = pd.concat([df, dfdf_clf_pred, dfdf_acc], axis =1)
                df.columns = ['retr_method', 'time', 'acc', 'em', 'f1', 'clf_pred', 'acc.1']
            else:
                dfdf_acc = pd.DataFrame([str(acc)])
                df = pd.concat([df, dfdf_acc], axis =1)
                df.columns = ['retr_method', 'time', 'acc', 'em', 'f1', 'acc.1']
            
        else:
            df = pd.DataFrame([[retr_method], [end-start],[sum(acc)/len(acc)], [metric.get_metric()['em']], [metric.get_metric()['f1']]]).T
            if retr_method == 'probing':
                dfdf_clf_pred = pd.DataFrame([str(retr_count_list)])    
                dfdf_acc = pd.DataFrame([str(acc)])
                df = pd.concat([df, dfdf_clf_pred, dfdf_acc], axis =1)
                df.columns = ['retr_method', 'time', 'acc', 'em', 'f1', 'clf_pred', 'acc.1']
            else:
                dfdf_acc = pd.DataFrame([str(acc)])
                df = pd.concat([df, dfdf_acc], axis =1)
                df.columns = ['retr_method', 'time', 'acc', 'em', 'f1', 'acc.1']
        save_path = "result"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df.to_csv(f'result/{args.ablation}_{_ds}_{retr_type}_{dataset_name}_{threshold}_{retr_method}_{savename_is_cot}_{tr_or_dev}_{steps_limit}.csv', index=False)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retr_method', type=str, default='') # probing, none, simple, skillrag, adaptive, flare, dragin, fix-length-retrieval, fix-sentence, linguistic
    parser.add_argument('--position', type=str, default='resid_post') # attn_out, resid_mid, mlp_out, resid_post
    parser.add_argument('--dataset_name', type=str, default='hotpotqa') # hotpotqa, nq, musique, 2wikimultihopqa, squad, trivia
    
    # dnese - squad, hotptoqa 메모리 부족 이슈
    # sparse - squad, hotptoqa 
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct') # google/gemma-2b meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen3-8B google/gemma-2-9b-it
    parser.add_argument('--tr_or_dev', type=str, default='dev') # train
    
    parser.add_argument('--ds', type=int, default=3) # 25,5, 75, 1000, 3
    parser.add_argument('--ablation', type=int, default=0) # 0-> 0 이후 모든 값 더하기
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--steps_limit', type=int, default=10000) # 1500 - 3 
    parser.add_argument('--max_retrieval_rounds', type=int, default=3)
    
    parser.add_argument('--is_sparse', action='store_true')
    parser.add_argument('--is_cot', action='store_true')
    parser.add_argument('--extracting_cot_qa', action='store_true')
    parser.add_argument('--extract_sep', action='store_true')
    parser.add_argument('--sep_number', type=int, default=3200)
    
    args = parser.parse_args()
    main(args)
    
#%%
    #%%
'''
###################################### make_dataset ##########################################################
python exp_rag.py --retr_method simple --is_sparse --tr_or_dev train --extracting_cot_qa --extract_sep --steps_limit 3200 --dataset_name trivia --is_cot --sep_number 0
python exp_rag.py --retr_method simple --is_sparse --tr_or_dev train --extracting_cot_qa --extract_sep --steps_limit 3200 --dataset_name hotpotqa --is_cot --sep_number 0
python exp_rag.py --retr_method simple --is_sparse --tr_or_dev train --extracting_cot_qa --extract_sep --steps_limit 3200 --dataset_name nq --is_cot --sep_number 0
python exp_rag.py --retr_method none --is_sparse --tr_or_dev train --extracting_cot_qa --extract_sep --steps_limit 3200 --dataset_name trivia --is_cot --sep_number 0
python exp_rag.py --retr_method none --is_sparse --tr_or_dev train --extracting_cot_qa --extract_sep --steps_limit 3200 --dataset_name hotpotqa --is_cot --sep_number 0
python exp_rag.py --retr_method none --is_sparse --tr_or_dev train --extracting_cot_qa --extract_sep --steps_limit 3200 --dataset_name nq --is_cot --sep_number 0

###################################### exp ##########################################################

python exp_rag.py --retr_method probing --steps_limit 500 --dataset_name nq --is_cot --is_sparse --model_id google/gemma-2b --ds 3
python exp_rag.py --retr_method probing --steps_limit 500 --dataset_name musique --is_cot --is_sparse --model_id google/gemma-2b --ds 3
python exp_rag.py --retr_method probing --steps_limit 500 --dataset_name hotpotqa --is_cot --is_sparse --model_id google/gemma-2b --ds 3
python exp_rag.py --retr_method probing --steps_limit 500 --dataset_name trivia --is_cot --is_sparse --model_id google/gemma-2b --ds 3
python exp_rag.py --retr_method probing --steps_limit 500 --dataset_name 2wikimultihopqa --is_cot --is_sparse --model_id google/gemma-2b --ds 3
'''
