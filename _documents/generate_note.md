# generate

## task required:
    + task = tasks.setup_task(cfg.task)
    + checkpoint_utils.load_model_ensemble
        + load_model_ensemble_and_task():
            + tasks.setup_task(cfg.task)
            + task.load_state_dict(state["task_state"]) (defined in FairseqTask, what's the point)
            + model = task.build_model(cfg.model)
    + task.load_dataset()
    + itr = task.get_batch_iterator() defined in FairseqTask waht is return? index??
        + require dataset setup
            + dataset.set_epoch(epoch) # not implemented in base class looking for somewhere else for an exampled implement
            + dataset.batch_by_size(): defined in fairseq_dataset
                + indices -> batch of samples
                + dataset.get_batch_shapes() 
                    + default returns None
                    + optional
                + dataset.num_tokens_vec(indices)
                + dataset.num_tokens(index)
                + data_utils.batch_by_size()
                + in data_utils, num_tokens_fn is usedif num_token_vec is not provided , so just implement num_tokens_fn is enough?
    + epoch_iter = iterators.EpochBatchIterator
        +  * it can call self.next_epoch_itr to get next itr
        + collate_fn=dataset.collater
        + in next_epoch_itr, call _get_iterator_for_epoch, then call torch.utils.data.DataLoader() to get itr
        + self.dataset.prefetch (optional) in _get_iterator_for_epoch()
    + generator = task.build_generator() implemented in fairseq_task
    + tokenizer = task.build_tokenizer(cfg.tokenizer) (implemented in encoder)
    + bpe = task.build_bpe(cfg.bpe)
    + prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]
    
# TODO:
how sequencegenerator generate words and how the model.forward_encoder /decoder works #756 762


# collater:
    + required dict keys: #212 generate
        + "id"
        + "target"
        + "net_input"
        + "src_tokens"
        + "constraints": optional
        + if it has "source", it should has "padding_mask"
    + task.inference_step()
# How it decode:

    hypos = task.inference_step() # ids
    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(hypo_tokens=hypo["tokens"], remove_bpe=...):
        hypo_str = tgt_dict.string(hypo_tokens, remove_bpe)
        hypo_tokens = tgt_dict.encode_line(hypo_str)
    detok_hypo_str = decode_fn(hypo_str)

    src_tokens: ->strip_pad->tgt_dict.string() (settings remove bpd /ingnore unk/post_process)-> decode_fn 
    decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x
+ for align_dict:
    + task.dataset(cfg.dataset.gen_subset).src.get_original_text()

# TODO

def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]

# model:
encoder_out = model.forward_encoder(net_input)
lprobs, avg_attn_scores = self.model.forward_decoder(
                tokens[:, : step + 1],
                encoder_outs,
                incremental_states,
                self.temperature)