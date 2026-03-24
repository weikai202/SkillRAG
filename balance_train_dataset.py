import pandas as pd
import argparse

RANDOM_STATE = 42
SUPPORTED_MODELS = [
    "google/gemma-2b",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen3-8B",
    "google/gemma-2-9b-it",
]
MODEL_SAVE_DIR = {
    "google/gemma-2b": "2b",
    "meta-llama/Meta-Llama-3-8B-Instruct": "8b",
    "Qwen/Qwen3-8B": "8b",
    "google/gemma-2-9b-it": "9b",
}


def get_paths(model_id: str, dataset_name: str):
    model_short = model_id.split("/")[-1]
    save_dir = MODEL_SAVE_DIR[model_id]
    input_path = f"dataset/{save_dir}/retrieval_qa_{model_short}_{dataset_name}_all_train_in3_.csv"
    output_path = f"dataset/{save_dir}/retrieval_qa_{model_short}_{dataset_name}_all_train_in3_balanced.csv"
    return input_path, output_path


def balance_one_model(model_id: str, dataset_name: str) -> None:
    input_path, output_path = get_paths(model_id, dataset_name)
    df = pd.read_csv(input_path)
    pos = df[df["acc"] == 1]
    neg = df[df["acc"] == 0]

    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(f"Cannot balance dataset for {model_id}: one class in 'acc' is empty.")

    n = min(len(pos), len(neg))
    pos_bal = pos.sample(n=n, random_state=RANDOM_STATE)
    neg_bal = neg.sample(n=n, random_state=RANDOM_STATE)

    balanced = pd.concat([pos_bal, neg_bal], axis=0).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    balanced.to_csv(output_path, index=False)
    print(f"Model: {model_id}")
    print(f"Saved balanced dataset: {output_path}")
    print(f"Original size: {len(df)}, Balanced size: {len(balanced)}")
    print(f"acc=1: {int((balanced['acc'] == 1).sum())}, acc=0: {int((balanced['acc'] == 0).sum())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/gemma-2b", choices=SUPPORTED_MODELS)
    parser.add_argument("--dataset_name", type=str, default="nq", choices=["trivia", "hotpotqa", "nq"])
    parser.add_argument("--all_models", action="store_true")
    args = parser.parse_args()

    if args.all_models:
        for model_id in SUPPORTED_MODELS:
            balance_one_model(model_id, args.dataset_name)
    else:
        balance_one_model(args.model_id, args.dataset_name)


if __name__ == "__main__":
    main()


### python balance_train_dataset.py --model_id google/gemma-2b
### python balance_train_dataset.py --model_id meta-llama/Meta-Llama-3-8B-Instruct
### python balance_train_dataset.py --model_id Qwen/Qwen3-8B
### python balance_train_dataset.py --model_id google/gemma-2-9b-it
