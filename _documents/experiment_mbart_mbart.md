# mbart25/50 with special tokens without entity, triple or position embedding:
+ setting:
    + aligned tokenizers in both sides (encoder and decoder)
        + bpe model: sentencepiece
        + dictionary: mbart50 dictionary with added tags:
            


# mbart25/50 without special tokens:



# mbart25/50 raw text:

# bart finetune 768 

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

