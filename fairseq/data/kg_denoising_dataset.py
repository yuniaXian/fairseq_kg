# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch

from . import FairseqDataset, data_utils




def collate(
    samples,
    pad_idx,
    eos_idx,
    vocab,
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
):
    assert input_feeding
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=None,  # use eos_idx of each sample instead of vocab.eos()
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor([s["source"].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s["target"]) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s["source"]) for s in samples)

    batch = {
        "id": id,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
        "nsentences": samples[0]["source"].size(0),
        "sort_order": sort_order,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens

    return batch


class KgDenoisingDataset(FairseqDataset):
    """
    A wrapper around TokenBlockDataset for BART dataset.

    Args:
        dataset (TokenBlockDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        mask_idx (int): dictionary index used for masked token
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
        seed: Seed for random number generator for reproducibility.
        args: argparse arguments.
    """

    def __init__(
        self,
        cur_lang_tag,
        dataset,
        sizes,
        vocab,
        mask_idx,
        mask_whole_words,
        mask_speical_tokens,
        shuffle,
        seed,
        args,
        eos=None,
        item_transform_func=None,
    ):
        self.cur_lang_tag = cur_lang_tag
        self.dataset = dataset

        self.sizes = sizes

        self.vocab = vocab
        self.shuffle = shuffle
        self.seed = seed
        self.mask_idx = mask_idx
        self.mask_whole_word = mask_whole_words
        self.mask_special_token = mask_speical_tokens
        self.whole_word_mask_mode = args.whole_word_mask_mode
        self.mask_ratio = args.mask
        self.random_ratio = args.mask_random
        self.insert_ratio = args.insert
        self.rotate_ratio = args.rotate
        self.permute_sentence_ratio = args.permute_sentences
        self.eos = eos if eos is not None else vocab.eos()
        self.item_transform_func = item_transform_func
        self.kg_tags = set(["[KG]", "[TEXT]", "[ENT]", "[TRIPLE]", "[PRED]", "[SUB]", "[SEP]", "<mask>"])
        self.lang_tags = set(['[ar_AR]','[cs_CZ]', '[de_DE]', '[en_XX]', '[es_XX]', \
                '[et_EE]', '[fi_FI]', '[fr_XX]', '[gu_IN]', '[hi_IN]', '[it_IT]', 
                '[ja_XX]', '[kk_KZ]', 
                '[ko_KR]', '[lt_LT]', '[lv_LV]', '[my_MM]', '[ne_NP]', '[nl_XX]', 
                '[ro_RO]', '[ru_RU]', '[si_LK]', '[tr_TR]', '[vi_VN]', '[zh_CN]', 
                '[af_ZA]', '[az_AZ]', '[bn_IN]', '[fa_IR]', '[he_IL]', '[hr_HR]', 
                '[id_ID]', '[ka_GE]', '[km_KH]', '[mk_MK]', '[ml_IN]', '[mn_MN]', 
                '[mr_IN]', '[pl_PL]', '[ps_AF]', '[pt_XX]', '[sv_SE]', '[sw_KE]', 
                '[ta_IN]', '[te_IN]', '[th_TH]', '[tl_XX]', '[uk_UA]', '[ur_PK]', 
                '[xh_ZA]', '[gl_ES]', '[sl_SI]'])




        if args.bpe != "gpt2":
            self.full_stop_index = self.vocab.eos()
        else:
            assert args.bpe == "gpt2"
            self.full_stop_index = self.vocab.index("13")

        self.replace_length = args.replace_length
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        if args.mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(f"invalid arg: mask-length={args.mask_length}")
        if args.mask_length == "subword" and args.replace_length not in [0, 1]:
            raise ValueError(f"if using subwords, use replace-length=1 or 0")

        self.mask_span_distribution = None
        if args.mask_length == "span-poisson":
            _lambda = args.poisson_lambda

            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            tokens = self.dataset[index]
            assert tokens[-1] == self.eos # TODO
            source, target = tokens, tokens.clone()

            # TODO add attr in init, add entities counts, add facts counts
            # only mask one part entity/pred/sub
            """
            self.mask_ent_ratio, self.mask_pred_ratio, self.mask_sub_ratio = \
                self.mask_kg_triples_ratio
            """
            
            # [en_XX][KG][ENT] sweet potato [TRIPLE] [PRED] ingrediment [SUB] xxx recipe [TRIPLE]
            # [en_XX] [ENT] ▁United ▁Kingdom [TRIPLE] [PRED] ▁description [SUB] ▁United ▁Kingdom [TRIPLE] [PRED] ▁a ssem bly [SUB] ▁A ston ▁Martin ▁V 8 [TRIPLE]
            # ent part: between [ENT] and [TRIPLE]
            # sub part: between [SUB] and [TRIPLE]
            # pred part: between [PRED] and [SUB]

            if self.mask_ratio > 0:
                source = self.add_whole_word_mask(source, self.mask_ratio)

            """
            if self.mask_ent_ratio > 0:
                pass
            if self.mask_pred_ratio > 0:
                pass
            if self.mask_sub_ratio > 0:
                pass
            
            """
            if self.permute_sentence_ratio > 0.0:
                source = self.permute_sentences(source, self.permute_sentence_ratio)

            if self.insert_ratio > 0:
                source = self.add_insertion_noise(source, self.insert_ratio)

            if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
                source = self.add_rolling_noise(source)
        # there can additional changes to make:
        if self.item_transform_func is not None:
            source, target = self.item_transform_func(source, target)

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.vocab)).all()
        assert source[0] == self.vocab.bos() or source[0] == self.vocab.index(self.cur_lang_tag)# TODO change why?
        assert source[-1] == self.eos # why  TODO
        return {
            "id": index,
            "source": source,
            "target": target,
        }

    def __len__(self):
        return len(self.dataset)

    
    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
            # the same as self.mask_whole_word[source]
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0 # eox TODO ?
        return is_word_start

    def special_tokens(self, source):
        if self.mask_special_token is not None:
            is_special_token = self.mask_special_token.gather(0, source)
        else:
            is_special_token = torch.ones(source.size())
        is_special_token[0] = 0
        is_special_token[-1] = 0 
        return is_special_token

    def add_whole_word_mask(self, source, p):
        is_word_start = self.word_starts(source)
        is_special_token = self.special_tokens(source)

        if self.whole_word_mask_mode == "word":
            is_word_start += is_special_token
        elif self.whole_word_mask_mode == "tag":
            is_word_start = is_special_token
        elif self.whole_word_mask_mode == "mixed":
            pass
        
        num_to_mask = int(math.ceil((is_word_start==1).sum()*p))
        num_inserts = 0
        if num_to_mask == 0:
            return source
        
        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0 # TODO ??
        word_starts = (is_word_start == 1).nonzero(as_tuple=False)
        #special_toekns = is_special_token.nonzero(as_tuple=False)
        # word_starts: indexes which are word starts
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        # indices: indexes to be masked
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(
                1, len(self.vocab), size=(mask_random.sum(),)
            )

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.vocab), size=(mask_random.sum(),)
                    )
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        1, len(self.vocab), size=(mask_random.sum(),)
                    )

                assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source
    
    def add_whole_ent_mask(self, source, p):
        # determine where is ent part
        ent_starts = source.index(self.ent)
        ent_ends = [source[s:].index(self.triple) for s in ent_starts]
        assert len(ent_starts) == len(ent_ends)
        ent_starts_ends = [[ent_starts[i], ent_ends[i]] for i in range(len(ent_starts))]
        # debug TOOD
        ent_to_data_index = np.stack(
        [
            np.arange(len(ent_starts)),  # starting index in dataset
            np.array(ent_starts_ends, dtype=np.compat.long),  # starting offset within starting index
        ],
        1, )
        ent_lst = ent_to_data_index[0]
        num_to_mask = int(math.ceil(len(ent_starts).float().sum() * p))
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            raise NotImplementedError
        else:
            indices = ent_lst[torch.randperm(ent_lst.size(0))[:num_to_mask]].squeeze(1)
        # entities to be masked

    
    def add_whole_propery_mask(self, source, p, mask_tags=False):
        # mask one class of property with ratio p
        pass

    def add_tag_mask_only(self, source, p, tag):
        pass
    
    def ratio_partition(self):
        pass
    
    def permute_sentences(self, source, p=1.0):
        full_stops = source == self.full_stop_index
        # Pretend it ends with a full stop so last span is a sentence
        full_stops[-2] = 1

        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero(as_tuple=False) + 2
        result = source.clone()

        num_sentences = sentence_ends.size(0)
        num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
        substitutions = torch.randperm(num_sentences)[:num_to_permute]
        ordering = torch.arange(0, num_sentences)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        # Ignore <bos> at start
        index = 1
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i > 0 else 1) : sentence_ends[i]]
            result[index : index + sentence.size(0)] = sentence
            index += sentence.size(0)
        return result

    
    def add_permuted_noise(self, tokens, p):
        num_words = len(tokens)
        num_to_permute = math.ceil(((num_words * 2) * p) / 2.0)
        substitutions = torch.randperm(num_words - 2)[:num_to_permute] + 1
        tokens[substitutions] = tokens[substitutions[torch.randperm(num_to_permute)]]
        return tokens

    def add_rolling_noise(self, tokens):
        offset = np.random.randint(1, max(1, tokens.size(-1) - 1) + 1)
        tokens = torch.cat(
            (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),
            dim=0,
        )
        return tokens

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(
            low=1, high=len(self.vocab), size=(num_random,)
        )

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return collate(
            samples, self.vocab.pad(), self.eos, self.vocab, pad_to_length=pad_to_length
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.sizes[indices], kind="mergesort")]

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
            hasattr(self.src, "supports_prefetch")
            and self.src.supports_prefetch
            and hasattr(self.tgt, "supports_prefetch")
            and self.tgt.supports_prefetch
        )