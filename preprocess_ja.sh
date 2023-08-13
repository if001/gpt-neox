#!/bin/sh
python tools/preprocess_data.py \
            --input ./data/mydataset.jsonl.zst \
            --output-prefix ./data/wiki_ja_en \            
            --vocab-file ./novelAI/tokenizer.model \
            --dataset-impl mmap \
            --tokenizer-type SPMTokenizer \
            --append-eod
