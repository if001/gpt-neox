# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from abc import ABC, abstractmethod
from multiprocessing import cpu_count

"""
This registry is for automatically downloading and extracting datasets.

To register a class you need to inherit the DataDownloader class, and provide name and url attributes, and (optionally)
the number of documents.

When done, add it to the DATA_DOWNLOADERS dict. The function process_data runs the pre-processing for the selected
dataset.
"""

GPT2_VOCAB_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json"
GPT2_MERGE_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt"


class DataDownloader(ABC):
    """Dataset registry class to automatically download / extract datasets"""

    def __init__(
        self,
        tokenizer_type=None,
        merge_file=None,
        vocab_file=None,
        data_dir=None,
        force_redownload=None,
        num_workers=None,
    ):
        if tokenizer_type is None:
            tokenizer_type = "GPT2BPETokenizer"
        if data_dir is None:
            data_dir = os.environ.get("DATA_DIR", "./data")
        if merge_file is None:
            merge_file = f"{data_dir}/gpt2-merges.txt"
        if force_redownload is None:
            force_redownload = False
        if vocab_file is None:
            if tokenizer_type == "GPT2BPETokenizer":
                vocab_file = f"{data_dir}/gpt2-vocab.json"
            elif tokenizer_type == "HFGPT2Tokenizer":
                vocab_file = "gpt2"
            elif tokenizer_type == "CharLevelTokenizer":
                pass
            else:
                assert vocab_file is not None, "No vocab file provided"
        if num_workers is None:
            num_workers = cpu_count()
        self._tokenizer_type = tokenizer_type
        self._merge_file = merge_file
        self._vocab_file = vocab_file
        self._data_dir = data_dir
        self._force_redownload = force_redownload
        self._num_workers = num_workers

    @property
    def base_dir(self):
        """base data directory"""
        return self._data_dir

    @property
    @abstractmethod
    def name(self):
        """name of dataset"""
        pass

    @property
    @abstractmethod
    def urls(self):
        """URLs from which to download dataset"""
        pass

    @property
    def tokenizer_type(self):
        """tokenizer type to use when tokenizing data"""
        return self._tokenizer_type

    @property
    def merge_file(self):
        """Merge file for tokenizer"""
        return self._merge_file

    @property
    def vocab_file(self):
        """Vocab file for tokenizer"""
        return self._vocab_file

    @property
    def num_workers(self):
        """Number of workers to use in preprocessing"""
        return self._num_workers

    @property
    def num_docs(self):
        """Number of documents in the dataset (if known)"""
        return None

    @property
    def ftfy(self):
        """Use ftfy (https://github.com/LuminosoInsight/python-ftfy) to fix text encodings"""
        return False

    def exists(self):
        """Checks if the dataset is present"""
        return os.path.isdir(f"{self.base_dir}/{self.name}")

    def download(self):
        """downloads dataset"""
        os.makedirs(os.path.join(self.base_dir, self.name), exist_ok=True)
        for url in self.urls:
            try:
                os_cmd = f"wget {url} -O {os.path.join(self.base_dir, self.name, os.path.basename(url))}"
                if os.system(os_cmd) != 0:
                    raise Exception(
                        f"Cannot download file at URL {url}: server may be down"
                    )
            except Exception as e:
                raise Exception(f"Download error: {e}")

    def tokenize(self):
        """tokenizes dataset"""
        parent_folder = os.path.join(self.base_dir, self.name)
        jsonl_filepath = ",".join(
            [os.path.join(parent_folder, os.path.basename(url)) for url in self.urls]
        )

        cmd = f"python tools/preprocess_data.py \
            --input {jsonl_filepath} \
            --output-prefix {parent_folder}/{self.name} \
            --vocab {self.vocab_file} \
            --dataset-impl mmap \
            --tokenizer-type {self.tokenizer_type} \
            --merge-file {self.merge_file} \
            --append-eod \
            --workers {self.num_workers} "

        if self.num_docs is not None:
            cmd += f"--num-docs {self.num_docs} "

        if self.ftfy:
            cmd += f"--ftfy "

        os.system(cmd)

    def prepare(self):
        if self._force_redownload:
            self.download()
        else:
            if not self.exists():
                self.download()

        self.tokenize()


class Enron(DataDownloader):
    name = "enron"
    urls = ["http://eaidata.bmk.sh/data/enron_emails.jsonl.zst"]
    num_docs = 517401


class PileSubset(DataDownloader):
    name = "pile_00"
    urls = ["https://the-eye.eu/public/AI/pile/train/00.jsonl.zst"]


class Pile(DataDownloader):
    name = "pile"
    urls = [
        f"https://the-eye.eu/public/AI/pile/train/{i:02}.jsonl.zst" for i in range(30)
    ]


class Github(DataDownloader):
    name = "github"
    urls = ["http://eaidata.bmk.sh/data/github_small.jsonl.zst"]


class ArXiv(DataDownloader):
    name = "arxiv"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/2020-09-08-arxiv-extracts-nofallback-until-2007-068.tar.gz"
    ]


class EuroParl(DataDownloader):
    name = "europarl"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/EuroParliamentProceedings_1996_2011.jsonl.zst"
    ]


class FreeLaw(DataDownloader):
    name = "freelaw"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst"
    ]


class NiH(DataDownloader):
    name = "nih"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/NIH_ExPORTER_awarded_grant_text.jsonl.zst"
    ]


class PubMed(DataDownloader):
    name = "pubmed"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/PMC_extracts.tar.gz"
    ]


class Books1(DataDownloader):
    name = "books1"
    urls = ["https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz"]


class Books3(DataDownloader):
    name = "books3"
    urls = ["https://the-eye.eu/public/AI/pile_preliminary_components/books3.tar.gz"]


class HackerNews(DataDownloader):
    name = "hackernews"
    urls = ["https://the-eye.eu/public/AI/pile_preliminary_components/hn.tar.gz"]
    num_docs = 373000


class OpenWebText2(DataDownloader):
    name = "openwebtext2"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar"
    ]
    num_docs = 17103000


class StackExchange(DataDownloader):
    name = "stackexchange"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/stackexchange_dataset.tar"
    ]


class UbuntuIRC(DataDownloader):
    name = "ubuntu_irc"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/ubuntu_irc_until_2020_9_1.jsonl.zst"
    ]


class YoutubeSubtitles(DataDownloader):
    name = "youtube_subtitles"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/yt_subs.jsonl.zst"
    ]


class C4(DataDownloader):
    name = "c4"
    urls = [
        f"https://the-eye.eu/eleuther_staging/c4/en/c4-train.{i:05}-of-01024.json.gz"
        for i in range(1024)
    ]


class C4OpenWebText(DataDownloader):
    name = "c4_openwebtext"
    urls = [
        f"https://the-eye.eu/eleuther_staging/c4/realnewslike/c4-train.{i:05}-of-00512.json.gz"
        for i in range(512)
    ]


class Enwik8(DataDownloader):
    name = "enwik8"
    urls = ["https://data.deepai.org/enwik8.zip"]


class WikiEn(DataDownloader):
    name = "wiki_en"
    urls = [        
        "https://dumps.wikimedia.org/other/cirrussearch/20230807/enwiki-20230807-cirrussearch-content.json.gz"
    ]


class WikiJa(DataDownloader):
    name = "wiki_ja"
    urls = [
        "https://dumps.wikimedia.org/other/cirrussearch/20230807/jawiki-20230807-cirrussearch-content.json.gz",        
    ]

class DataDownloaderWithHF(DataDownloader):
    def __init__(self, hf_repo_ids = [], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hf_repo_ids = hf_repo_ids

    def download(self):
        super().download()
        from huggingface_hub import snapshot_download 
        save_dir = os.path.join(self.base_dir, self.name)
        for repo_id in self.hf_repo_ids:            
            snapshot_download(repo_id=repo_id, revision="main", allow_patterns="*.jsonl", local_dir=save_dir, repo_type='dataset')

class WikiOSCARJa(DataDownloaderWithHF):
    name = "wiki_oscar_ja"
    urls = [
        "https://dumps.wikimedia.org/other/cirrussearch/20230807/jawiki-20230807-cirrussearch-content.json.gz",        
    ]
    hf_repo_ids = [
        'if001/oscar_2023_filtered'
    ]

class HFSnapshotDownloader(DataDownloader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def hf_repo_ids(self):
        pass

    def download(self):        
        from huggingface_hub import snapshot_download        
        save_dir = os.path.join(self.base_dir, self.name)
        for repo_id in self.hf_repo_ids:
            print('save to', save_dir)
            allow_patterns = None
            if 'if001/oscar_2023_filtered' == repo_id:
                allow_patterns="*.jsonl.zst"
            if 'if001/aozorabunko-clean-sin' == repo_id:
                allow_patterns="*.jsonl.gz"
            snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns, local_dir=save_dir, repo_type="dataset")


class OSCARJa(HFSnapshotDownloader):
    name = "oscar_ja"
    urls = [""]
    hf_repo_ids = ['if001/oscar_2023_filtered']

class AozoraJa(HFSnapshotDownloader):
    name = "aozora_ja"
    urls = [""]
    hf_repo_ids = ['if001/aozorabunko-clean-sin']

class HFDataDownloader(DataDownloader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def hf_repo_ids(self):
        pass

    def download(self):        
        from datasets import load_dataset
        save_dir = os.path.join(self.base_dir, self.name)
        for repo_id in self.hf_repo_ids:
            ds = load_dataset(repo_id)
            name = repo_id.split('/')[0]
            save_path = f'{save_dir}/{name}.jsonl'
            print('save to', save_path)
            ds['train'].to_json(save_path, force_ascii=False)


class IzumiFullDataset(HFDataDownloader):
    name = "izumi_full_dataset"
    urls = [""]
    hf_repo_ids = [
        "izumi-lab/wikipedia-ja-20230720",
        "izumi-lab/wikipedia-en-20230720",
        "izumi-lab/wikinews-ja-20230728"
    ]

class IzumiWikiJaDataset(HFDataDownloader):
    name = "izumi_wiki_ja_dataset"
    urls = [""]
    hf_repo_ids = [
        "izumi-lab/wikipedia-ja-20230720",
    ]

class IzumiWikiEnDataset(HFDataDownloader):
    name = "izumi_wiki_en_dataset"
    urls = [""]
    hf_repo_ids = [
        "izumi-lab/wikipedia-en-20230720",        
    ]

class IzumiWikiNewsJaDataset(HFDataDownloader):
    name = "izumi_wiki_news_dataset"
    urls = [""]
    hf_repo_ids = [
        "izumi-lab/wikinews-ja-20230728"        
    ]    

def maybe_download_gpt2_tokenizer_data(tokenizer_type, data_dir):
    if tokenizer_type is None or tokenizer_type == "GPT2BPETokenizer":
        GPT2_VOCAB_FP = f"{data_dir}//gpt2-vocab.json"
        GPT2_MERGE_FP = f"{data_dir}/gpt2-merges.txt"
        if not os.path.isfile(GPT2_VOCAB_FP):
            os.system(f"wget {GPT2_VOCAB_URL} -O {GPT2_VOCAB_FP}")
        if not os.path.isfile(GPT2_MERGE_FP):
            os.system(f"wget {GPT2_MERGE_URL} -O {GPT2_MERGE_FP}")


DATA_DOWNLOADERS = {
    "pass": "pass",
    "enron": Enron,
    "pile_subset": PileSubset,
    "pile": Pile,
    "github": Github,
    "arxiv": ArXiv,
    "europarl": EuroParl,
    "freelaw": FreeLaw,
    "nih": NiH,
    "pubmed": PubMed,
    "books1": Books1,
    "books3": Books3,
    "hackernews": HackerNews,
    "openwebtext2": OpenWebText2,
    "stackexchange": StackExchange,
    "ubuntu_irc": UbuntuIRC,
    "youtube_subtitles": YoutubeSubtitles,
    "c4": C4,
    "c4_openwebtext": C4OpenWebText,
    "enwik8": Enwik8,
    'wiki_en': WikiEn,
    'wiki_ja': WikiJa,
    'oscar_ja': OSCARJa,
    'wiki_oscar_ja': WikiOSCARJa,
    'aozora_ja': AozoraJa,
    'izumi_dataset': IzumiFullDataset,
    'izumi_wiki_ja_dataset': IzumiWikiJaDataset,
    'izumi_wiki_en_dataset': IzumiWikiEnDataset,
    'izumi_wiki_news_dataset': IzumiWikiNewsJaDataset
}


def prepare_dataset(
    dataset_name: str,
    tokenizer_type: str = None,
    data_dir: str = None,
    vocab_file: str = None,
    merge_file: str = None,
    force_redownload: bool = None,
    num_workers: int = None,
):
    """
    Downloads + tokenizes a dataset in the registry (dataset_name) and saves output .npy files to data_dir.
    """
    if data_dir is None:
        data_dir = os.environ.get("DATA_DIR", "./data")
    os.makedirs(data_dir, exist_ok=True)
    maybe_download_gpt2_tokenizer_data(tokenizer_type, data_dir)
    DownloaderClass = DATA_DOWNLOADERS.get(dataset_name.lower(), None)
    if DownloaderClass is None:
        raise NotImplementedError(
            f'Dataset "{dataset_name}" not recognized - please choose from {list(DATA_DOWNLOADERS.keys())}'
        )
    elif DownloaderClass == "pass":
        # pass on building dataset (for unit tests)
        pass
    else:
        num_workers = 1 if dataset_name == "enwik8" else num_workers
        d = DownloaderClass(
            tokenizer_type=tokenizer_type,
            vocab_file=vocab_file,
            merge_file=merge_file,
            data_dir=data_dir,
            force_redownload=force_redownload,
            num_workers=num_workers,
        )
        d.prepare()
