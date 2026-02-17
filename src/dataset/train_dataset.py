import random
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from src.arguments import DataArguments
from src.utils import build_record

class TrainDataset(Dataset):
    """
    Dataset for training which handles both query and passage data.
    Loads dataset and optional corpus from the provided paths/configurations.
    """

    def __init__(self,
                 data_args: DataArguments,
                 trainer=None,
                 dataset_name=None,
                 corpus_name=None,
                 dataset_path=None,
                 corpus_path=None,
                 corpus_assets_path=None):
        self.data_args = data_args
        self.trainer = trainer

        # Load training data
        self.train_data = load_dataset(
            self.data_args.dataset_name if dataset_name is None else dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path if dataset_path is None else dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            num_proc=self.data_args.num_proc,
        )

        # Load corpus if provided
        if self.data_args.corpus_name is None and corpus_name is None:
            self.corpus = None
        else:
            self.corpus = load_dataset(
                self.data_args.corpus_name if corpus_name is None else corpus_name,
                self.data_args.corpus_config,
                data_files=self.data_args.corpus_path if corpus_path is None else corpus_path,
                split=self.data_args.corpus_split,
                cache_dir=self.data_args.dataset_cache_dir,
                num_proc=self.data_args.num_proc,
            )
        
        # for video we use assets_path to load the video
        self.corpus_assets_path = corpus_assets_path if corpus_assets_path is not None else self.data_args.assets_path

        # create a map between docid and index
        self.docid_to_index = {}
        if self.corpus is not None:
            corpus_ids = self.corpus.select_columns(['docid'])
            docids = corpus_ids['docid']
            self.docid_to_index = {docid: index for index, docid in enumerate(tqdm(docids))}

        self.query_modalities = self.data_args.resolve_modalities(is_query=True)
        self.passage_modalities = self.data_args.resolve_modalities(is_query=False)

    def set_trainer(self, trainer):
        """Sets the trainer for the dataset."""
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def _get_info_from_docid(self, docid):
        """
        Retrieves document information from the corpus given a docid.
        Returns:
            tuple: (formatted_text, image, video, audio)
        """
        document_info = self.corpus[self.docid_to_index[docid]]
        assert document_info['docid'] == docid

        content_text = document_info.get('text', '')
        if 'title' in document_info:
            content_text = document_info['title'] + ' ' + content_text
        content_text = content_text.strip()

        record = build_record(
            docid,
            content_text,
            document_info.get('image', None),
            document_info.get('video', None),
            document_info.get('audio', None),
            self.passage_modalities,
            self.data_args.passage_prefix,
            self.corpus_assets_path,
        )
        return record

    def _getitem_legacy(self, group, epoch, _hashed_seed):
        formatted_query = build_record(
            group.get('query_id', group.get('id', None)),
            group.get('query', None),
            None,
            None,
            None,
            self.query_modalities,
            self.data_args.query_prefix,
        )

        formatted_documents = []
        # Select positive document
        selected_positive = group['positive_passages'][(_hashed_seed + epoch) % len(group['positive_passages'])]
        positive_text = (selected_positive['title'] + ' ' + selected_positive['text']) if 'title' in selected_positive else selected_positive['text']
        positive_text = positive_text.strip()
        formatted_documents.append(
            build_record(
                selected_positive.get('docid', selected_positive.get('id', None)),
                positive_text,
                None,
                None,
                None,
                self.passage_modalities,
                self.data_args.passage_prefix,
            )
        )

        # Select negative documents
        negative_size = self.data_args.train_group_size - 1
        if len(group['negative_passages']) < negative_size:
            selected_negatives = random.choices(group['negative_passages'], k=negative_size)
        elif self.data_args.train_group_size == 1:
            selected_negatives = []
        else:
            offset = epoch * negative_size % len(group['negative_passages'])
            selected_negatives = list(group['negative_passages'])
            random.Random(_hashed_seed).shuffle(selected_negatives)
            selected_negatives = selected_negatives * 2
            selected_negatives = selected_negatives[offset: offset + negative_size]

        for negative in selected_negatives:
            negative_text = (negative['title'] + ' ' + negative['text']) if 'title' in negative else negative['text']
            negative_text = negative_text.strip()
            formatted_documents.append(
                build_record(
                    negative.get('docid', negative.get('id', None)),
                    negative_text,
                    None,
                    None,
                    None,
                    self.passage_modalities,
                    self.data_args.passage_prefix,
                )
            )

        return {"query": formatted_query, "passages": formatted_documents}

    def _getitem_new(self, group, epoch, _hashed_seed):
        formatted_query = build_record(
            group.get('query_id', group.get('id', None)),
            group.get('query_text', group.get('query', None)),
            group.get('query_image', group.get('image', None)),
            group.get('query_video', group.get('video', None)),
            group.get('query_audio', group.get('audio', None)),
            self.query_modalities,
            self.data_args.query_prefix,
            self.data_args.assets_path,
        )

        formatted_documents = []
        positive_document_ids = group['positive_document_ids']
        negative_document_ids = group['negative_document_ids']

        # Select positive document id
        selected_positive_docid = positive_document_ids[(_hashed_seed + epoch) % len(positive_document_ids)]
        formatted_documents.append(
            self._get_info_from_docid(selected_positive_docid)
        )

        # Select negative document ids
        negative_size = self.data_args.train_group_size - 1
        if self.data_args.train_group_size == 1:
            selected_negative_docids = []
        elif len(negative_document_ids) == 0:
            # No negatives available, return empty list
            selected_negative_docids = []
        elif len(negative_document_ids) < negative_size:
            selected_negative_docids = random.choices(negative_document_ids, k=negative_size)
        else:
            offset = epoch * negative_size % len(negative_document_ids)
            selected_negative_docids = list(negative_document_ids)
            random.Random(_hashed_seed).shuffle(selected_negative_docids)
            selected_negative_docids = selected_negative_docids * 2
            selected_negative_docids = selected_negative_docids[offset: offset + negative_size]

        for neg_docid in selected_negative_docids:
            formatted_documents.append(
                self._get_info_from_docid(neg_docid)
            )

        return {"query": formatted_query, "passages": formatted_documents}

    def __getitem__(self, item):
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(item + self.trainer.args.seed)

        # Handling the legacy format with 'positive_passages'
        if 'positive_passages' in group:
            return self._getitem_legacy(group, epoch, _hashed_seed)
        # Handling the new format
        else:
            return self._getitem_new(group, epoch, _hashed_seed)