# Train an LLM on WhatsApp chats

A small GPT trained from scratch on exported WhatsApp chats, in notebooks: clean the chats, train a BPE tokenizer, pretrain a transformer, then fine-tune it to reply in a chat. The last series (8_x) repeats the process on a larger Moroccan Arabic (Darija) text dataset.

**Status:** the notebooks can't be run as they are. See [Known issues](#known-issues).

## Requirements

- Python 3.12 (what it was written on; not tested on other versions)
- A GPU helps from notebook 4 on; most of them fall back to CPU

## Install

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`minbpe/` is a vendored copy of Karpathy's tokenizer library. The notebooks do `sys.path.append('..')` to find it, so start Jupyter from `notebooks/`.

## Data

Not in the repo (`data/` and `output/` are gitignored). The notebooks expect:

```
data/private/*.txt             WhatsApp "export chat" files (without media)
data/private/fine_tuning.txt   one chat for the fine-tuning set
data/AtlaSetCombined.txt       written by 8_1
data/supervised_fine_tuning.json
output/                        tokenizer, encoded data, checkpoints
```

## Notebooks

Run in order. 4 to 7 are the WhatsApp series, 8_x the Darija series.

| Notebook | What it does |
|---|---|
| 1_Data_cleaning | Parses the chat exports, drops system messages, media, links and emails, merges everything into `output/combined_text.txt` |
| 2_BPE | Trains a 1024-token `BasicTokenizer`, adds the special tokens |
| 3_Transformer_model | Defines the GPT (6 layers, 6 heads, 384 embedding, 256 context) |
| 4_Train_model_all_batches, 4_train_model_random_batches | Two ways of pretraining on the combined text |
| 5_FineTuningDataset | Groups consecutive messages per sender, wraps them in `<\|startoftext\|>` / `<\|separator\|>` / `<\|endoftext\|>`, saves JSON |
| 6_1, 6_2 | Fine-tune on the chat data with no earlier context |
| 7_ParameterEfficeintFineTuning | Same, with LoRA |
| 8_1 | Downloads `atlasia/Atlaset` from Hugging Face into one text file |
| 8_2, 8_3 | Trains a 16,384-token `RegexTokenizer`, encodes the text to `.npy` |
| 8_4 | Pretrains a larger model (8 layers, 512 embedding, 1024 context) |
| 8_5_1, 8_5_2 | Fine-tunes on Q&A and on conversations, with and without loss masking |

## Known issues

- Notebooks 4 to 8 import `GPTLanguageModel` from a `transformer.model` module (`transformers.model` in 4_Train_model_all_batches, `GPTlanguageModel` in 6_1) that isn't in the repo. The class is defined in notebook 3 and has to be exported by hand. Notebook 7 also imports `get_lora_model` and `print_trainable_parameters` from `transformers.lora`, which aren't defined anywhere, so 7 can't run.
- Hardcoded paths: `C:\Users\DELL\Desktop\trainwhatasppllm\...` in notebooks 1 to 4, and `"path"` placeholders in 2, 4_train_model_random_batches, 5, 6_1 and 8_2. Edit them to your own.
- Notebook 4_Train_model_all_batches opens `combined_text_txt` (missing dot).
- Notebook 5 stores the grouped text under `"messages"` but reads `"message"` later, which looks like a `KeyError`. Not run, from reading only.
- The chat data is private, so the WhatsApp series can't be reproduced from the repo alone.

What was checked: `minbpe` imports and round-trips text with `BasicTokenizer` and `RegexTokenizer`. Nothing in `notebooks/` was run.
