
## ConcatDataset(FairseqDataset)
    + self.datasets: list of datasets
    + self.sample_ratios: [1,2,..] integers. 2 for 2 copies of corresponding dataset 
    + self.cumulative_sizes : 1619, 2000 -> [1619, 3619]
    + self.real_sizes: [1619, 2000]
    + __getitem__: idx -> find the corresponding dataset and idx in this dataset: return self.datasets[dataset_idx][sample_idx]
    + num_tokens: return np.max(self.size(index))
    + sizes: return list of sizes of all samples
    + size(idx): return size of sample with idx
    + 

## ResamplingDataset(BaseWrapperDataset):
    + use random seed, size_ratio, and epoch info to set resampling indices array, and get item w.r.t that array
    + self._cur_indices, self._cur_epoch
    + self.seed
    + self.actual_size = np.ceil(len(dataset) * self.size_ratio).astype(int)
    + __getitem__: return self.dataset[self._cur_indices.array[index]]
    + self.set_epoch(epoch) return array

## SortDataset(BaseWrapperDataset):
    +  self.ordered_indices(): np.lexsort(sort_order)
        a = [1,5,1,4,3,4,4] # First column
        b = [9,4,0,4,0,2,1] # Second column
        ind = np.lexsort((b,a)) # Sort by a, then by b
        ind
        array([2, 0, 4, 6, 5, 3, 1])


## multilingual_denoisingDataset:
    + folders:
        + data_path/split for langs is None
        + data_path/lang/split for langs is not None
        eg. dataset_denoising/kgtext_wikidata/en_XX/valid, dataset_denoising/kgtext_wikidata/zh_CN/valid, etc
    + lang_datasets = [lang_dataset_1, lang_dataset_2,...], lang_dataset_k corresponds to lang_k, one dataset for one lang (for each split)
    + for split == "train":
        lang_dataset_k = ResamplingDataset(lang_dataset_k)
    + for split == "valid":
        {"split": self.dataset. "split_lang1": dataset_1, "split_lang2": dataset_2, ... }
        eg. self.datasets = {"valid": self.dataset. "valid_en_XX": lang_dataset_1, "valid_zh_CN": lang_dataset_2, ... }
    + dataset: concatenate(lang_datasets)
    + self.datasets: dict:
        eg. self.datasets = {"valid": dataset. "valid_en_XX": lang_dataset_1, "valid_zh_CN": lang_dataset_2, ... , "train": dataset}
    + "train" dataset is loaded in checkpoint_utils.load_checkpoint() -> trainer.get_train_iterator()
 
 ## rainer.get_train_iterator():
    + load "train" split dataset
    + return batch_iterator = self.task.get_batch_iterator():
        + # create mini-batches with given size constraints
            batch_sampler = dataset.batch_by_size()
        + epoch_iter = iterators.EpochBatchIterator(.., batch_sampler,...): call torch.utils.data.DataLoader inside
    + Get an iterator that yields batches of data from the given dataset.

    

        