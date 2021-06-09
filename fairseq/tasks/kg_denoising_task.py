# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    DenoisingDataset,
    KgDenoisingDataset,
    Dictionary,
    PrependTokenDataset,
    ResamplingDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask1, get_special_token_mask
from fairseq.tasks import register_task

from .denoising import DenoisingTask


logger = logging.getLogger(__name__)


@register_task("kg_multilingual_denoising")
class KgMultilingualDenoisingTask(DenoisingTask):
    @staticmethod
    def add_args(parser):
        DenoisingTask.add_args(parser)
        parser.add_argument(
            "--multilang-sampling-alpha",
            type=float,
            default=1.0,
            help="smoothing alpha for sample ratios across multiple datasets",
        )
        parser.add_argument("--add-lang-token", default=False, action="store_true")
        parser.add_argument(
            "--langs", type=str, help="language ids we are considering", default=None
        )
        parser.add_argument(
            "--no-whole-word-mask-langs",
            type=str,
            default="",
            metavar="N",
            help="languages without spacing between words dont support whole word masking",
        )

        parser.add_argument(
            "--whole_word_mask_mode",
            type=str,
            default="word",
            metavar="N",
            help="word: whole word masking will not mask tags; mixed: whole word masking will mask tags or words; tags: whole word masking will only mask tags",
        )

        parser.add_argument(
            "--property_mask_mode",
            type=bool,
            default=False,
            metavar="N",
            help="to be implemented",
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task."""
        paths = args.data.split(":")
        assert len(paths) > 0
        dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))

        data_path = paths[0]
        if args.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ]
            )
        else:
            languages = args.langs.split(",")

        if args.add_lang_token:
            for lang in languages:
                dictionary.add_symbol("[{}]".format(lang))

        logger.info("dictionary: {} types".format(len(dictionary)))
        if not hasattr(args, "shuffle_instance"):
            args.shuffle_instance = False
        return cls(args, dictionary)

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.dictionary = dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx = self.dictionary.add_symbol("<mask>")
        self.langs = args.langs
        self.args = args

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by languages. This helps low resource
        languages by upsampling them.
        """
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob ** self.args.multilang_sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(":")
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        # /local/home/xianjiay/efs-storage/data-bin/dataset_denoising/webnlg
        split_path = os.path.join(data_path, split)

        if self.langs is None:
            languages = sorted( # get all the dirs folders under data_path
                [
                    name
                    for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))
                ]
            )
        else:
            languages = self.langs.split(",")
            for name in languages:
                p = os.path.join(data_path, name) # for example '/local/home/xianjiay/efs-storage/data-bin/dataset_denoising/webnlg/en_XX'
                assert os.path.exists(p), "data not found: {}".format(p)

        logger.info("Training on {0} languages: {1}".format(len(languages), languages))
        logger.info(
            "Language to id mapping: ", {lang: id for id, lang in enumerate(languages)}
        )

        mask_whole_words = get_whole_word_mask1(self.args, self.dictionary)
        # mask_whole_words: tensor([1,1,1,1,0,0,0,....,1,1]) start of word: 1 else: 0
        mask_speical_tokens = get_special_token_mask(self.args, self.dictionary) # TODO GPU dtype?

        language_without_segmentations = self.args.no_whole_word_mask_langs.split(",")
        lang_datasets = []
        for language in languages:
            split_path = os.path.join(data_path, language, split)

            dataset = data_utils.load_indexed_dataset( # if impl is raw, it will add eos, so remove eos of bped dataset
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=True, # TODO change
            )
            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )

            end_token = (
                self.source_dictionary.eos()
            )

            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample - 2,  # one less for <s>
                pad=self.source_dictionary.pad(), # pad index
                eos=end_token, # eos index
                break_mode=self.args.sample_break_mode, #?? TODO: check
            )
            logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))


            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            """
            if name == "kg2kg":
                dataset = PrependTokenDataset(dataset, self.source_dictionary.index("[KG]"))
            elif name == "text2text":
                dataset = PrependTokenDataset(dataset, self.source_dictionary.index("[TEXT]"))
            elif name == "kg2text":
                pass
            else:
                dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

            dataset = PrependTokenDataset(dataset, self.source_dictionary.index("[{}]".format(language)))
            """

            #dataset = AppendTokenDataset(dataset, end_token)

            lang_mask_whole_words = (
                mask_whole_words
                if language not in language_without_segmentations
                else None
            )
            lang_tag = "[{}]".format(language)

            lang_dataset = KgDenoisingDataset(
                lang_tag,
                dataset,
                dataset.sizes,
                self.dictionary,
                self.mask_idx,
                lang_mask_whole_words,
                mask_speical_tokens,
                shuffle=self.args.shuffle_instance,
                seed=self.seed,
                args=self.args,
                eos=self.dictionary.eos()
            )
            lang_datasets.append(lang_dataset)

        print(lang_dataset[0])
        print(lang_dataset[2000])

        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        logger.info(
            "loaded total {} blocks for all languages".format(
                int(dataset_lengths.sum()),
            )
        )
        if split == self.args.train_subset:
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(dataset_lengths)
            logger.info(
                "Sample probability by language: {}".format(
                    {
                        lang: "{0:.4f}".format(sample_probs[id])
                        for id, lang in enumerate(languages)
                    }
                )
            )
            size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
            logger.info(
                "Up/Down Sampling ratio by language: {}".format(
                    {
                        lang: "{0:.2f}".format(size_ratio[id])
                        for id, lang in enumerate(languages)
                    }
                )
            )

            resampled_lang_datasets = [
                ResamplingDataset(
                    lang_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            dataset = ConcatDataset(
                resampled_lang_datasets,
            )
        else:
            dataset = ConcatDataset(lang_datasets)
            lang_splits = [split]
            for lang_id, lang_dataset in enumerate(lang_datasets):
                split_name = split + "_" + languages[lang_id] # eval_en_XX
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset

            if split in self.args.valid_subset:
                self.args.valid_subset = self.args.valid_subset.replace(
                    split, ",".join(lang_splits)
                )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))

        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )