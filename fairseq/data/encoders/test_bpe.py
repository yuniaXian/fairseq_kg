from fairseq.data.encoders.gpt2_bpe import GPT2BPEConfig, GPT2BPE


cfg = GPT2BPEConfig()
encoder_json_path = "/home/ubuntu/efs-storage/tokenizer/gpt2/bpe/encoder.json"
vocab_bpe_path = "/home/ubuntu/efs-storage/tokenizer/gpt2/bpe/vocab.bpe"
cfg.gpt2_encoder_json = "/home/ubuntu/efs-storage/tokenizer/gpt2/bpe/encoder.json"
cfg.gpt2_vocab_bpe = "/home/ubuntu/efs-storage/tokenizer/gpt2/bpe/vocab.bpe"

#setattr(cfg, gpt2_encoder_json, encoder_json_path)
#setattr(cfg, gptgpt2_vocab_bpe, vocab_bpe_path)

bpe = GPT2BPE(cfg)
bpe.bpe.encode("We can use the existing fairseq-train command-line tool for this, making sure to specify our new Model architecture")
print(1)



