from datasets import load_dataset
from torch.utils.data import Dataset
from src.arguments import DataArguments
from src.utils import build_record

import logging
logger = logging.getLogger(__name__)

class EncodeDataset(Dataset):
    """
    Dataset for encoding.
    Loads data and optionally shards it for distributed processing.
    """

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        self.encode_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.query_path if self.data_args.encode_is_query else self.data_args.corpus_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            num_proc=self.data_args.num_proc,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )

        self.is_query = self.data_args.encode_is_query
        self.modalities = self.data_args.resolve_modalities(self.is_query)

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item):
        content = self.encode_data[item]
        if self.data_args.encode_is_query:
            content_id = content.get('query_id') or content.get('id') or content.get('query-id', None)
            if content_id is None:
                raise KeyError(f"Could not find query_id, id, or query-id in content. Available keys: {list(content.keys())}")
            content_text = content.get('query_text', content.get('query', ''))
            content_image = content.get('query_image', None)
            content_video = content.get('query_video', None)
            content_audio = content.get('query_audio', None)
        else:
            content_id = content['docid'] if 'docid' in content else content['id']
            content_text = content.get('text', '')
            if 'title' in content:
                content_text = content['title'] + ' ' + content_text
            content_text = content_text.strip()
            content_image = content.get('image', None)
            content_video = content.get('video', None)
            content_audio = content.get('audio', None)

        record = build_record(
            content_id,
            content_text,
            content_image,
            content_video,
            content_audio,
            self.modalities,
            self.data_args.query_prefix if self.is_query else self.data_args.passage_prefix,
            self.data_args.assets_path,
        )
        return record