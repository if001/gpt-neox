#!/bin/sh
python prepare_data.py -d ./data \
-t SPMTokenizer \
--vocab-file ./novelAI/tokenizer.model \
wiki_ja_en
