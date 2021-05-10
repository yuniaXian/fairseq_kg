
--multilang-sampling-alpha 1.0
#smoothing alpha for sample ratios across multiple datasets",
--add-lang-token False
--langs "en_XX" type=str, #language ids we are considering"
--no-whole-word-mask-langs str languages without spacing between words dont support whole word masking
--tokens-per-sample 512
--sample-break-mode complete_doc mode for breaking sentence
--mask 0.0 fraction of words/subwords that will be masked
--mask-random 0.0 instead of using [MASK], use random token this often
--insert 0.0 insert this percentage of additional random tokens
--permute 0.0 take this proportion of subwords and permute them
--rotate 0.5 rotate this proportion of inputs
--poisson-lambda 3.0 randomly shuffle sentences for this proportion of inputs",
--permute-sentences 0.0 shuffle this proportion of sentences in all inputs
--mask-length choices=["subword", "word", "span-poisson"]
--replace-length -1 when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)",
--max-source-position 1024 max number of tokens in the source sequence
--max-target-positions
--shorten-method "none"  choices=["none", "truncate", "random_crop"], help="if not none, shorten sequences that exceed --tokens-per-sample",
--shorten-data-split-list comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)


