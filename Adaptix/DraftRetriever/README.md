## DraftRetriever

DraftRetriever is an integral component of [ADED](https://github.com/liuxukun2000/ADED), a Retrieval-Based Speculative Decoding method that accelerates large language model (LLM) decoding without fine-tuning, using an adaptive draft-verification process. It dynamically adjusts to token probabilities with a tri-gram matrix representation and employs Monte Carlo Tree Search (MCTS) to balance exploration and exploitation, producing accurate drafts quickly. ADED significantly speeds up decoding while maintaining high accuracy, making it ideal for practical applications.

### Installation

```sh
pip install draftretriever
```

#### Prerequisites:
If the provided wheel files are not compatible with your system, ensure you have Rust installed:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
```



**Build from source**

Using `install.sh` or:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
maturin build --release --strip -i python3.9 
pip install [.whl]
```

### Example

#### Generate Tri-gram Matrix
```python
import draftretriever
from transformers import AutoTokenizer
from tqdm import tqdm
import json

tokenizer = AutoTokenizer.from_pretrained(model_path)


datastore_path = './datastore_chat_large.idx'
writer = draftretriever.Writer(
    file_path=datastore_path,
    vocab_size=tokenizer.vocab_size,
)

dataset_path = "datastore/ShareGPT_V4.3_unfiltered_cleaned_split.json"
assert dataset_path is not None, "please download the dataset from https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered"
dataset = json.load(open(dataset_path))
total_length = len(dataset)
print("number of samples: ", total_length)
for conversations in tqdm(dataset, total=total_length):
    for sample in conversations['conversations']:
        token_list = tokenizer.encode(sample['value'])
        writer.add_entry(token_list)

writer.finalize()
```
#### Search
```python
import draftretriever

datastore = draftretriever.Reader(index_file_path=datastore_path)
retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = datastore.search(token_list, choices=max_num_draft)
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgement
The main framework is from [REST](https://github.com/FasterDecoding/REST)


