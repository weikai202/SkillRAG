import pandas as pd


INPUT_PATH = "dataset/2b/retrieval_qa_gemma-2b_all_train_in3_.csv"
OUTPUT_PATH = "dataset/2b/retrieval_qa_gemma-2b_all_train_in3_balanced.csv"
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    pos = df[df["acc"] == 1]
    neg = df[df["acc"] == 0]

    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Cannot balance dataset: one class in 'acc' is empty.")

    n = min(len(pos), len(neg))
    pos_bal = pos.sample(n=n, random_state=RANDOM_STATE)
    neg_bal = neg.sample(n=n, random_state=RANDOM_STATE)

    balanced = pd.concat([pos_bal, neg_bal], axis=0).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    balanced.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved balanced dataset: {OUTPUT_PATH}")
    print(f"Original size: {len(df)}, Balanced size: {len(balanced)}")
    print(f"acc=1: {int((balanced['acc'] == 1).sum())}, acc=0: {int((balanced['acc'] == 0).sum())}")


if __name__ == "__main__":
    main()
