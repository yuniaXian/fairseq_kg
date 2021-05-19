# mbart25/50 with tags without entity, triple or position embedding:
+ setting:
    + aligned tokenizers in both sides (encoder and decoder)
        + bpe model: sentencepiece
        + dictionary: mbart50 dictionary with added tags:
            

+ finetune: 54.02 54.28
+ train from scratch: Generate test with beam=5: BLEU4 = 47.51, 78.7/59.2/45.2/35.9 (BP=0.906, ratio=0.910, syslen=46654, reflen=51247)
# mbart25/50 without special tokens:
+ finetune: 54.25
+ train from scratch: Generate test with beam=5: BLEU4 = 46.95, 79.5/59.8/45.8/36.3 (BP=0.886, ratio=0.892, syslen=45701, reflen=51247)


# 768 embedding size /6 layers/ 8 heads transformer:
+ with tags: 47.10 with 3e-5 lr/46.49 with 1e-04 lr
+ without tags: 46.81 with 3e-5 lr

# using gpt2 encoder:


# denoising task:
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
    + implement (for triples):
        + mask the whole entity or predicate or subject part, special tokens are not masked
            +  [KG] [lang] [TRIPLE] [ENT] {sweet potato} [pred] ingrediment [SUB] French dishes [TRIPLE]
   
            +  [KG] [lang] [TRIPLE] [ENT] sweet potato [TRIPLE] [pred] ingrediment [SUB] French dishes [TRIPLE]

        + randomly permute the entity, pedicate or subject to other word/words:
            +  [KG] [lang] [TRIPLE] [ENT] {bear} [pred] ingrediment [SUB] French dishes [TRIPLE]

        + mask random numbers of whole words in one part and only in one part
            + [KG] [lang] [TRIPLE] [ENT] {sweet} potato[pred] ingrediment [SUB] {French} dishes [TRIPLE]
        
        + mask special tokens only (set a ratio to determine how many should be masked) and a probability to shffle the positions:
            + [KG] [lang] [TRIPLE]  {[pred]} ingrediment {[SUB]} French dishes {[ENT]} sweet potato[TRIPLE]
            + [KG] [lang] [TRIPLE] [ENT] sweet potato [pred] ingrediment [SUB] French dishes [TRIPLE]

        [KG] [lang]       -> [TEXT] [lang] sentence
                          -> [KG] [lang] triples
        [TEXT][lang]

    + rotate positions for supervised training:
        + rotate positions of different triples -> text
            + [KG] [lang] [TRIPLE]  {[pred]} ingrediment {[SUB]} French dishes {[ENT]} sweet potato[TRIPLE]
                -> [TEXT] [lang] sweet potatoes are xx ingrediment of French dishes.

            + 
