



# mbart25/50 with tags without entity, triple or position embedding:
    + setting:
        + aligned tokenizers in both sides (encoder and decoder)
            + bpe model: sentencepiece
            + dictionary: mbart50 dictionary with added tags:
                

## finetune: bleu score from fairseq: 1st: 54.02 2nd: 54.25

NIST score = 11.6247  BLEU score = 0.6525 for system "tst"
------------------------------------------------------------------------

Individual N-gram scoring
        1-gram   2-gram   3-gram   4-gram   5-gram   6-gram   7-gram   8-gram   9-gram
        ------   ------   ------   ------   ------   ------   ------   ------   ------
 NIST:  7.7902   2.3927   0.8174   0.4116   0.2128   0.1143   0.0648   0.0388   0.0256  "tst"

 BLEU:  0.9086   0.7369   0.5908   0.4745   0.3790   0.3014   0.2397   0.1922   0.1540  "tst"
------------------------------------------------------------------------
Cumulative N-gram scoring
        1-gram   2-gram   3-gram   4-gram   5-gram   6-gram   7-gram   8-gram   9-gram
        ------   ------   ------   ------   ------   ------   ------   ------   ------
 NIST:  7.7902   10.1829   11.0003   11.4118   11.6247   11.7389   11.8037   11.8425   11.8681  "tst"

 BLEU:  0.9006   0.8111   0.7276   0.6525   0.5843   0.5225   0.4669   0.4174   0.3732  "tst"
MT evaluation scorer ended on 2021 May 19 at 20:03:49

Removing temp directory
SCORES:
==============
BLEU: 0.6525
NIST: 11.6247
METEOR: 0.4707
ROUGE_L: 0.7513
CIDEr: 4.5431

## train from scratch: Generate test with beam=5: BLEU4 = 47.51, 78.7/59.2/45.2/35.9 (BP=0.906, ratio=0.910, syslen=46654, reflen=51247)

# mbart25/50 with lang tags + without special tokens:
done
# mbart25/50 raw text:
?
# tansformer #6 768 / #6 512 #6 1024

# data without lang tags + tags /notags

## finetune: 54.25 54.45
+ train from scratch: Generate test with beam=5: BLEU4 = 46.95, 79.5/59.8/45.8/36.3 (BP=0.886, ratio=0.892, syslen=45701, reflen=51247)

NIST score = 11.6727  BLEU score = 0.6580 for system "tst"

------------------------------------------------------------------------

Individual N-gram scoring
        1-gram   2-gram   3-gram   4-gram   5-gram   6-gram   7-gram   8-gram   9-gram
        ------   ------   ------   ------   ------   ------   ------   ------   ------
 NIST:  7.7911   2.4143   0.8299   0.4211   0.2162   0.1158   0.0650   0.0389   0.0258  "tst"

 BLEU:  0.9129   0.7430   0.5958   0.4789   0.3823   0.3044   0.2436   0.1943   0.1542  "tst"

------------------------------------------------------------------------
Cumulative N-gram scoring
        1-gram   2-gram   3-gram   4-gram   5-gram   6-gram   7-gram   8-gram   9-gram
        ------   ------   ------   ------   ------   ------   ------   ------   ------
 NIST:  7.7911   10.2054   11.0354   11.4565   11.6727   11.7886   11.8536   11.8925   11.9184  "tst"

 BLEU:  0.9056   0.8170   0.7334   0.6580   0.5893   0.5272   0.4716   0.4217   0.3768  "tst"
MT evaluation scorer ended on 2021 May 19 at 20:13:06


    Removing temp directory
    SCORES:
    ==============
    BLEU: 0.6580
    NIST: 11.6727
    METEOR: 0.4696
    ROUGE_L: 0.7536
    CIDEr: 4.5512


# 768 embedding size /6 layers/ 8 heads transformer:
+ with tags: 47.10 with 3e-5 lr/46.49 with 1e-04 lr
+ without tags: 46.81 with 3e-5 lr BLEU4 = 46.81, 77.3/58.3/44.6/35.4 (BP=0.906, ratio=0.911, syslen=46664, reflen=51247)

NIST score = 10.0371  BLEU score = 0.5603 for system "tst"

------------------------------------------------------------------------

Individual N-gram scoring
        1-gram   2-gram   3-gram   4-gram   5-gram   6-gram   7-gram   8-gram   9-gram
        ------   ------   ------   ------   ------   ------   ------   ------   ------
 NIST:  6.8405   2.0098   0.6752   0.3390   0.1726   0.0933   0.0531   0.0307   0.0214  "tst"

 BLEU:  0.8544   0.6709   0.5290   0.4169   0.3273   0.2571   0.2023   0.1585   0.1229  "tst"

------------------------------------------------------------------------
Cumulative N-gram scoring
        1-gram   2-gram   3-gram   4-gram   5-gram   6-gram   7-gram   8-gram   9-gram
        ------   ------   ------   ------   ------   ------   ------   ------   ------
 NIST:  6.8405   8.8503   9.5255   9.8645   10.0371   10.1304   10.1835   10.2142   10.2356  "tst"

 BLEU:  0.8028   0.7114   0.6313   0.5603   0.4970   0.4407   0.3908   0.3464   0.3066  "tst"
MT evaluation scorer ended on 2021 May 19 at 20:18:02

    Removing temp directory
    SCORES:
    ==============
    BLEU: 0.5603
    NIST: 10.0371
    METEOR: 0.4013
    ROUGE_L: 0.6877
    CIDEr: 3.5738

+ without tags:
NIST score = 9.9905  BLEU score = 0.5600 for system "tst"

------------------------------------------------------------------------

Individual N-gram scoring
        1-gram   2-gram   3-gram   4-gram   5-gram   6-gram   7-gram   8-gram   9-gram
        ------   ------   ------   ------   ------   ------   ------   ------   ------
 NIST:  6.8025   2.0044   0.6752   0.3364   0.1720   0.0923   0.0506   0.0289   0.0195  "tst"

 BLEU:  0.8563   0.6759   0.5357   0.4216   0.3284   0.2554   0.1982   0.1551   0.1196  "tst"

------------------------------------------------------------------------
Cumulative N-gram scoring
        1-gram   2-gram   3-gram   4-gram   5-gram   6-gram   7-gram   8-gram   9-gram
        ------   ------   ------   ------   ------   ------   ------   ------   ------
 NIST:  6.8025   8.8069   9.4821   9.8185   9.9905   10.0828   10.1334   10.1623   10.1818  "tst"

 BLEU:  0.7975   0.7085   0.6303   0.5600   0.4962   0.4389   0.3879   0.3428   0.3026  "tst"
MT evaluation scorer ended on 2021 May 19 at 20:33:04

Removing temp directory
SCORES:
==============
BLEU: 0.5600
NIST: 9.9905
METEOR: 0.4003
ROUGE_L: 0.6853
CIDEr: 3.5368

# implement copygate
# implement kgpt embedding:

<<<<<<< HEAD
# bart finetune 768
=======
# using gpt2 encoder:
>>>>>>> 685e28719c6c0172d2559147edf243c318df4a32


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
