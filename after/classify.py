import torch
import tiktoken
from previous_chapters import GPTModel, load_weights_into_gpt
from data_prepare import SpamDataset
from pathlib import Path
import pandas as pd


def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

def create_balanced_dataset(df):
    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df

def classify_review(
        text,model,tokenizer,device,max_length = None,
        pad_token_id = 50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]

    input_ids = input_ids[:min(
        max_length , supported_context_length
    )]

    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(input_ids , device = device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:,-1,:]
        predicted_label = torch.argmax(logits,dim=-1).item()
        return 'spam' if predicted_label == 1 else 'not spam'

if __name__ == "__main__":
    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )

    import argparse


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Finetune a GPT model for classification"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("This flag runs the model in test mode for internal testing purposes. "
              "Otherwise, it runs the model as it is used in the chapter (recommended).")
    )
    args = parser.parse_args()
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

     #拆分 训练 测试 验证集
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    #数据加载
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SpamDataset(
        csv_file="train.csv",
        max_length=None,
        tokenizer=tokenizer
    )

    val_dataset = SpamDataset(
        csv_file="validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    test_dataset = SpamDataset(
        csv_file="test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)



    if args.test_mode:
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 120,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2
        }
        model = GPTModel(BASE_CONFIG)
        model.eval()
        device = "cpu"

        # Code as it is used in the main chapter
    else:
        CHOOSE_MODEL = "gpt2-small (124M)"
        INPUT_PROMPT = "Every effort moves"

        BASE_CONFIG = {
            "vocab_size": 50257,  # Vocabulary size
            "context_length": 1024,  # Context length
            "drop_rate": 0.0,  # Dropout rate
            "qkv_bias": True  # Query-key-value bias
        }

        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
            f"Dataset length {train_dataset.max_length} exceeds model's context "
            f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
            f"`max_length={BASE_CONFIG['context_length']}`"
        )

        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        from after.gpt_download import download_and_load_gpt2

        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

        model = GPTModel(BASE_CONFIG)
        load_weights_into_gpt(model, params)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #数据加载
    tokenizer = tiktoken.get_encoding("gpt2")

    print(classify_review(
        text_1, model, tokenizer, device, max_length=train_dataset.max_length
    ))

    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    print(classify_review(
        text_2, model, tokenizer, device, max_length=train_dataset.max_length
    ))