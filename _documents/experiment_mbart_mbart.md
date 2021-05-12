# mbart25/50 with special tokens:
+ setting:
    + aligned tokenizers in both sides (encoder and decoder)
        + bpe model: sentencepiece
        + dictionary: mbart50 dictionary with added tags:
            + special tokens:
                + bos: <s>
                + eos: </s>
                + unk: <unk>
                + pad: <pad>
            + added tokens to annotate triples, and parts of triples:
                + triple: object predicate subject
                + -> [ENT] object [TRIPLE] [pred] predicate [SUB] subject [TRIPLE]
                + multiple triples in one data sample: object1 predicate1 subject1, predicate2 subject2, (shared object)
                + -> [ENT] object1 [TRIPLE] [pred] predicate1 [SUB] subject1 [TRIPLE], [TRIPLE] [pred] predicate2 [SUB] subject2 [TRIPLE].
                + -> tokenized ids
            + 


# mbart25/50 without special tokens:



# mbart25/50 raw text:


# bart finetune

