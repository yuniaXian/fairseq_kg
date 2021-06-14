- [````:](#)
- [pytorch and tensor:](#pytorch-and-tensor)
  - [tensor.new([...])](#tensornew)
  - [torch.cat, torch.stack](#torchcat-torchstack)
  - [`tensor.clone(), tensor.detach()`](#tensorclone-tensordetach)
- [TokenBlockDataset(FairseqDataset):](#tokenblockdatasetfairseqdataset)
- [AppendTokenDataset(BaseWrapperDataset):](#appendtokendatasetbasewrapperdataset)
- [DenoisingDataset(FairseqDataset):](#denoisingdatasetfairseqdataset)

# getitem in denoising:
    `__getitem__(index)` ->
    ```
    BaseWrapperDataset:
        def __getitem__(self, index):
            return self.dataset[index]
    ConcatDataset(FairseqDataset):
        def __getitem__(self, idx):
            dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
            return self.datasets[dataset_idx][sample_idx]
        
        def _get_dataset_and_sample_index(self, idx: int):
            ...
            return dataset_idx, sample_idx
    DenoisingDataset(FairseqDataset):
        def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            tokens = self.dataset[index]
    AppendTokenDataset(BaseWrapperDataset):
        def __getitem__(self, idx):
            item = self.dataset[idx]
    PrependTokenDataset(BaseWrapperDataset):
        def __getitem__(self, idx):
            item = self.dataset[idx]
    TokenBlockDataset(FairseqDataset):
        def __getitem__(self, index):
            start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
        

    ```
# pytorch and tensor:
## tensor.new([...])
    This will create a tensor of Gaussian noise, the same shape and data type as a Variable X:
## torch.cat, torch.stack
    ```
    >>> a=torch.Tensor([1,2,3])
    >>> torch.stack((a,a)).size()
    torch.size(2,3)
    >>> torch.cat((a,a)).size()
    torch.size(6)
    ```
## `tensor.clone(), tensor.detach()`
    [link](https://blog.csdn.net/winycg/article/details/100813519)
    
    | Markdown               | new/shared memory | Still in computation graph |
    | ---------------------- | ----------------- | -------------------------- |
    | **x.clone()**          | new               | *true*                     |
    | **x.detach()**         | shared            | *false*                    |
    | **x.clone().detach()** | new               | *false*                    |
##  torhc.randperm():
    + Returns a random permutation of integers from $0$ to $n - 1$.
# TokenBlockDataset(FairseqDataset):
+ 返回三个重要值：_sizes, block_to_dataset_index, slice_indices
+ `self._block_to_dataset_index`：
    + [datalen, 3]
    + index -> start_ds_idx, start_offset, end_ds_idx
    + given index -> corresponding original block of data
    + 5 -> [5, 0, 5]
+ `self._sizes`:
    + PlasmaArray object
+ `self.sizes`:
    + num of tokens of each sample
    + index -> num of tokens of the index sample
    + tensor [48, 30, 64 ,...] 
+ `self._slice_indices`:
    + PlasmaArray object
    + the start idx and end idx for each sample (?? new or original) according to the total numbers of tokens
    + _slice_indices.array: $$[[0,48], [48, 78], [78, 142], ....]$$
+ `__getitem__`:
    ```
    start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
    buffer = torch.cat(
        [self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
    )
    slice_s, slice_e = self.slice_indices[index]
    length = slice_e - slice_s
    s, e = start_offset, start_offset + length
    item = buffer[s:e]
    ```
    + index -> start_ds_idx, start_offset, end_ds_idx
    + get involved part of dataset into buffer
    + get true length w.r.t given index sample
    + get item in buffer w.r.t to true length



    
# AppendTokenDataset(BaseWrapperDataset):
```
def __getitem__(self, idx):
    item = self.dataset[idx]
    if self.token is not None:
        item = torch.cat([item, item.new([self.token])])
    return item
```

# DenoisingDataset(FairseqDataset):
parameter: 
`mask_whole_words = get_whole_word_mask(self.args, self.dictionary)`
+ get_whole_word_mask(args, dictionary)
```
def get_whole_word_mask(args, dictionary):
    bpe = encoders.build_bpe(args)
    if bpe is not None:


        def is_beginning_of_word(i):
            if i < dictionary.nspecial:
                # special elements are always considered beginnings
                return True
            tok = dictionary[i]
            if tok.startswith("madeupword"):
                return True
            try:
                return bpe.is_beginning_of_word(tok)
            except ValueError:
                return True

        mask_whole_words = torch.ByteTensor(
            list(map(is_beginning_of_word, range(len(dictionary))))
        )
        return mask_whole_words
    return None
```


```
def __init__(self):
    self.mask_whole_word = mask_whole_words


def __getitem__(self, index):
    with data_utils.numpy_seed(self.seed, self.epoch, index):
        tokens = self.dataset[index]
        assert tokens[-1] == self.eos
        source, target = tokens, tokens.clone()

        ...
        if self.mask_ratio > 0:
            source = self.add_whole_word_mask(source, self.mask_ratio)

def add_whole_word_mask(self, source, p):
    is_word_start = self.word_starts(source)
    num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
    num_inserts = 0
    ....
    else:
        lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio


    + get is_word_start: tensor: 1-> is_start/0->is_not_start for each subword in the sample: [0, 0, 1, 1, ...]
    + get num_to_mask (# of total words * mask_ration)
    + word_starts: get all the indexes in the sample which is word start -> word_starts
    + indices: get a random list of indexes to mask
    + mask_random: use a random vocab word to replace

    source_length = source.size(0)
    assert source_length - 1 not in indices
    to_keep = torch.ones(source_length, dtype=torch.bool)
    is_word_start[-1] = 255
    ...
    else:
        # keep index, but replace it with [MASK]
        source[indices] = self.mask_idx
        source[indices[mask_random]] = torch.randint(
            1, len(self.vocab), size=(mask_random.sum(),)
        )
    ...
    else:
        # masking any subward if its start subwords are masked
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
                    1, len(self.vocab), size=(mask_random.sum(),))
    
    source = source[to_keep]
    return source

```
# bpe
if x in ["<unk>", "<s>", "</s>", "<pad>"]:
    # special elements are always considered beginnings
    # HACK: this logic is already present in fairseq/tasks/masked_lm.py
    # but these special tokens are also contained in the sentencepiece
    # vocabulary which causes duplicate special tokens. This hack makes
    # sure that they are all taken into account.
    return True
return x.startswith("\u2581")

```



