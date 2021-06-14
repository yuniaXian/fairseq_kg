import logging
from typing import Dict, List, Optional, Tuple, Any
from fairseq import utils, dataclass
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceConfig
from fairseq.data.encoders.gpt2_bpe import GPT2BPEConfig
from omegaconf import MISSING, OmegaConf, II
from dataclasses import dataclass, field
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE

logger = logging.getLogger(__name__)

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

@dataclass
class KgptConfig(FairseqDataclass):
    pad_token_id: int = field(default=50263, metadata={"help": "."})
    sos_token_id: int = field(default=50262, metadata={"help": "."})
    eos_token_id: int = field(default=50261, metadata={"help": "."})
    vocab_size: int = field(default=50264, metadata={"help": "."})
    hidden_size: int = field(default=768, metadata={"help": "."})

    max_entity_embeddings: int = field(default=30, metadata={"help": "."})
    max_triple_embeddings: int = field(default=20, metadata={"help": "."})
    max_position_embeddings: int = field(default=1024, metadata={"help": "."})

    layer_norm_eps: float = field(default=0.1, metadata={"help": "."})
    src_length: int = field(default=760, metadata={"help": "."})
    tgt_length: int = field(default=760, metadata={"help": "."})
    positionwise_copy_prob: bool = field(default=False, metadata={"help": "."})
    knowledge_file: str = field(default="knowledge-full.json", metadata={"help": "."})

    encoder_layerdrop: float = field(default=0.1, metadata={"help": "encoder_layerdrop"})
    decoder_layerdrop: float = field(default=0.1, metadata={"help": "decoder_layerdrop"})

    encoder_embed_dim: int = field(default=768, metadata={"help": "."})
    decoder_embed_dim: int = field(default=768, metadata={"help": "."})

    decoder_embed_path: str = field(default="", metadata={"help": "."})
    encoder_embed_path: str = field(default="", metadata={"help": "."})

    pretrained_encoder_file: str = field(default="model_ep14.pt", metadata={"help": "."}) # True
    pretrained_decoder_file: str = field(default="model_ep14.pt", metadata={"help": "."}) # True

    load_pretrained_encoder: bool = field(default=False, metadata={"help": "."}) # True
    load_pretrained_decoder: bool = field(default=False, metadata={"help": "."}) # True

    encoder_ffn_embed_dim: int = field(default=3072, metadata={"help": "."})
    decoder_ffn_embed_dim: int = field(default=3072, metadata={"help": "."})

    encoder_layers: int = field(default=6, metadata={"help": "."})
    encoder_attention_heads: int = field(default=8, metadata={"help": "."})

    decoder_layers: int = field(default=6, metadata={"help": "."})
    decoder_attention_heads: int = field(default=9, metadata={"help": "."})


@dataclass
class TransformerConfig(FairseqDataclass):
    #max_source_positions: Optional[int] = II("task.max_source_positions")
    #max_target_positions: Optional[int] = II("task.max_target_positions")

    max_source_positions: int = field(
        default=1024, metadata={"help": "maximum source position in positional embedding"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "maximum target position in positional embedding"}
    )
    # activation functions
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )

    # embeddings
    # encoder embeddings:
    encoder_embed_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to pre-trained encoder embedding"}
    )
    encoder_embed_dim: int = field(
        default=512, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "encoder embedding dimension for FFN"}
    )
    # decoder embeddings:
    decoder_embed_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to pre-trained decoder embedding"}
    )
    decoder_input_dim: int = field(
        default=512, metadata={"help": "decoder input dimension"}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_output_dim: int = field(
        default=512, metadata={"help": "decoder output dimension"}
    )

    # attentions
    # encoder attentions
    encoder_layers: int = field(
        default=6, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_attention_heads: int = field(
        default=8, metadata={"help": "num encoder attention heads"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    encoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each encoder block"}
    )
    encoder_learned_pos: bool = field(
        default=False, metadata={"help": "use learned positional embeddings in the encoder"}
    )
    encoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={"help": "which layers to *keep* when pruning as a comma-separated list"}
    )
    # decoder attentions
    decoder_layers: int = field(
        default=6, metadata={"help": "num encoder layers in the transformer"}
    )
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num encoder attention heads"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each encoder block"}
    )
    decoder_learned_pos: bool = field(
        default=False, metadata={"help": "use learned positional embeddings in the encoder"}
    )
    decoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={"help": "which layers to *keep* when pruning as a comma-separated list"}
    )

    # other embed_dim transformer configs
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    share_all_embeddings: bool = field(
        default=False,
        metadata={"help": 'share encoder, decoder and output embeddings'
                          ' (requires shared dictionary and embed dim)'}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    adaptive_input: bool = field(
        default=False, metadata={"help": "if set, uses adaptive input"}
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    no_cross_attention: bool = field(
        default=False,
        metadata={"help": "do not perform cross-attention"},
    )
    cross_self_attention: bool = field(
        default=False,
        metadata={"help": "perform cross+self-attention"},
    )
    # config for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )
    # config for Fully Sharded Data Parallel (FSDP) training
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": (
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        }
    )


@dataclass
class Kg2textModelConfig(FairseqDataclass):
    n_layers: int = field(default=6, metadata={"help": "number of layers, default: 6"})
    encoder_type: str = field(default="kgpt", metadata={"help": "encoder type: kgpt(default)/mbart50"})
    decoder_type: str = field(default="mbart50", metadata={"help": "decoder type: kgpt/mbart50(default)"})
    load_pretrained_encoder: bool = field(default=False, metadata={"help": "True/False: load pretrained encoder model file"})
    load_pretrained_decoder: bool = field(default=False, metadata={"help": "True/False: load pretrained decoder model file"})
    pretrained_encoder_file: str = field(default="encoder_kgpt.pt", metadata={"help": "specify encoder model file"})
    pretrained_decoder_file: str = field(default="decoder_mbart50.pt", metadata={"help": "specify decoder model file"})
    checkpoint_file: str = field(default="", metadata={"help": "speicify checkpoint model"})
    share_all_embeddings: bool = field(default=False, metadata={"help":""})
    encoder_embed_dim: int = field(default=768, metadata={"help": "encoder embedding dimension, default: kgpt: 768"})
    decoder_embed_dim: int = field(default=1024, metadata={"help": "decoder embedding dimension, default: mbart50: 1024"})
    encoder_embed_file: str = field(default="", metadata={"help": "encoder embedding layer file"})
    decoder_embed_file: str = field(default="", metadata={"help": "decoder embedding layer file"})
    save_encoder_file: str = field(default="", metadata={"help": "encoder saved file name"})
    save_decoder_file: str = field(default="", metadata={"help": "decoder saved file name"})
    save_checkpoint_file: str = field(default="", metadata={"help": "checkpoint saved file name"})
    use_copy_gate: bool = field(default=True, metadata={"help": "use copy gate?"})
    #copy_para_from_kgpt_enc: bool = field(default=False, metadata={"help": ""})
    #save_kgpt_encoder: bool = field(default=False, metadata={"help": ""})
    #save_kgpt_embed_tokens: bool = field(default=False, metadata={"help": ""})

    kgpt: KgptConfig = KgptConfig()
    mbart50: TransformerConfig = TransformerConfig()


@dataclass
class Kg2textTokenConfigSeperate(FairseqDataclass):

    src_tokenizer: str = field(default="gpt2", metadata={"help": "encoder_tokenizer: default is GPT2"})
    bpe_tokenizer_src: SentencepieceConfig = SentencepieceConfig()
    vocab_file_src: str = field(default="MISSING", metadata={"help": "for encoder, path to fairseq dictionary file"})

    prepend_src_lang_tag: bool = field(default=False, metadata={"help": "prepend src lang tag"})
    src_max: int = field(default=640, metadata={"help": "max length of data of encoder input"})
    src_max_length: int = field(default=100000, metadata={"help": "max length, what it is?"})
    src_lang_tag_template: str = field(default="[{}]", metadata={"help": "src: lang tag template"})
    src_enc_type: str = field(default="seq", metadata={"help": "type of encoder"})
    src_lang: str = field(default="en", metadata={"help": "language of source"})

    tgt_tokenizer: str = field(default="gpt2", metadata={"help": "decoder_tokenizer: default is GPT2"})
    bpe_tokenizer_tgt: SentencepieceConfig = SentencepieceConfig()
    vocab_file_tgt: str = field(default="MISSING", metadata={"help": "for decoder, path to fairseq dictionary file"})

    prepend_tgt_lang_tag: bool = field(default=False, metadata={"help": "prepend src lang tag"})
    tgt_max: int = field(default=72, metadata={"help": "max length of data of decoder input"})
    tgt_max_length: int = field(default=100000, metadata={"help": "max length, what it is?"})
    tgt_lang_tag_template: str = field(default="[{}]", metadata={"help": "tgt: lang tag template"})
    # dec_enc_type: str = field(default="seq", metadata={"help": "type of encoder"})
    tgt_lang: str = field(default="en", metadata={"help": "language of target"})

    lower_case: bool = field(default=False, metadata={"help": "lower case for src and tgt"})

@dataclass
class Kg2textMbart50Config(FairseqDataclass):
    tokenizer: str = field(default="mbart50", metadata={"help": "tokenizer type"})
    bpe_tokenizer: SentencepieceConfig = SentencepieceConfig()
    vocab_file: str = field(default="dict.mbart50.txt", metadata={"help": "dictionary file"})

@dataclass
class KgptGpt2Config(FairseqDataclass):
    tokenizer_dir: str = field(default="GPT2_tokenizer", metadata={"help": "."})

@dataclass
class Kg2textTokenConfig(FairseqDataclass):
    tokenizer_src: str = field(default="kgpt_gpt2", metadata={"help":"."})
    tokenizer_tgt: str = field(default="sentencepiece", metadata={"help":"."})
    vocab_file_src: str = field(default="dict.txt", metadata={"help": "."})
    vocab_file_tgt: str = field(default="dict.mbart50.txt", metadata={"help": "."})
    fairseq_gpt2: GPT2BPEConfig = GPT2BPEConfig()
    sentenpiece: SentencepieceConfig = SentencepieceConfig()
    kgpt_gpt2: KgptGpt2Config = KgptGpt2Config()


@dataclass
class Kg2textDataSetConfig(FairseqDataclass):
    # TODO: maybe just fix the directory structure to force it to be relative to II("task.data")
    # II(): dataset task.dataset
    fairseq_gpt2: GPT2BPEConfig = GPT2BPEConfig()
    sentenpiece: SentencepieceConfig = SentencepieceConfig()
    kgpt_gpt2: KgptGpt2Config = KgptGpt2Config()

    option: str = field(default="train", metadata={"help": "."}) # == task.option
    split: str = field(default="train", metadata={"help": "."})
    dataset: str = field(default="webnlg", metadata={"help": "."})
    tokenizer_src: str = field(default="kgpt_gpt2", metadata={"help": "source tokenizer type"})
    vocab_file_src: str = field(default="dict.txt", metadata={"help": "."})
    #bpe_tokenizer_src: GPT2BPEConfig = GPT2BPEConfig()
    #vocab_file_src: str = field(default= "dict.gpt2.json", metadata={"help": "source dictionary file"})

    tokenizer_tgt: str = field(default="mbart50", metadata={"help": "target tokenizer type"})
    vocab_file_tgt: str = field(default="dict.mbart50.txt", metadata={"help": "."})
    #bpe_tokenizer_tgt: SentencepieceConfig = SentencepieceConfig()
    #vocab_file_tgt: str = field(default="dict.mbart50.json", metadata={"help": "target dictionary file"})

    # dataset: str = field(default="wikidata", metadata={"help": "choose dataset"})
    # dataset: Optional[str] = II("task.dataset")
    num_workers: int = field(default=8, metadata={"help": "number of workers"})
    batch_size: int = field(default=64, metadata={"help": "batch size"})
    #train_file_path: str = field(default="", metadata={"help": "kg triples training dataset path relative to which to the project"})
    train_file: str = field(default="", metadata={"help": "train subset file name"})
    test_file: str = field(default="", metadata={"help": "val subset file name"})
    eval_file: str = field(default="", metadata={"help": "test subset file name"} )

    src_length: int = field(default=760, metadata={"help": "."})
    tgt_length: int = field(default=50, metadata={"help": "."})

    # multilingual related
    src_lang_tag_template: str = field(default="[{}]", metadata={"help": "."})
    tgt_lang_tag_template: str = field(default="[{}]", metadata={"help": "."})
    src_lang: str = field(default="", metadata={"help": "."})
    tgt_lang: str = field(default="", metadata={"help": "."})
    prepend_src_lang_tag: bool = field(default=False, metadata={"help": "."})
    prepend_tgt_lang_tag: bool = field(default=False, metadata={"help": "."})

    # with added tags
    # kgpt: enc/dec + lang_tags / mbart50: dec + special tags
    src_wtags: bool = field(default=True, metadata={"help": "."})
    tgt_wtags: bool = field(default=True, metadata={"help": "."})
    # Kgpt seq related
    max_entity: int = field(default=50, metadata={"help": "."})
    max_fact: int = field(default=50, metadata={"help": "."})
    percent: float = field(default=1.0, metadata={"help": ""})

    forbid_duplicate_relation: bool = field(default=True, metadata={"help": "."})
    lower_case: bool = field(default=False, metadata={"help": "."})
    encoder_arch: str = field(default="sequence", metadata={"help": "."})
    knowledge_file: str = field(default="", metadata={"help": "."})

    #max_entity: int = field(default=12, metadata={"help": "number of workers"})
    #max_fact: int = field(default=8, metadata={"help": "number of workers"})

    #forbid_duplicate_relation: bool = field(default=True, metadata={"help": "forbid_duplicate_relation..??"})
    #knowledge_path: str = field(default="/knowledge-full.json",
    #                           metadata={"help": "for wikidata training only, augmented entities knowledge"})
    #shuffle: bool = field(
    #    default=True,
    #    metadata={"help": "if set, shuffle dataset samples before batching"},
    #)

@dataclass
class Kg2textTaskConfig(FairseqDataclass):
    option: str = field(default="train", metadata={"help": "[train, pretrain, finetune, eval, test, challenge]"})
    split: str = field(default="train", metadata={"help": "[train/eval/test]"})
    dataset: str= field(default="webnlg", metadata={"help": "name of dataset: [webnlg, wikidata, wikibio]"})
    encoder_type: str = field(default="kgpt", metadata={"help": "encoder type: kgpt(default)/mbart50"})
    decoder_type: str = field(default="mbart50", metadata={"help": "decoder type: kgpt/mbart50(default)"})
    # load_data_from: str = field(default="", metadata={"help": "path the data directory"})
    # percent: float = field(default=1.0, metadata={"help": "percentage of data"})
    test: Kg2textDataSetConfig = Kg2textDataSetConfig()
    webnlg: Kg2textDataSetConfig = Kg2textDataSetConfig() # on tokenizer and preparing dataloader
    wikidata: Kg2textDataSetConfig = Kg2textDataSetConfig()
    wikibionlg: Kg2textDataSetConfig = Kg2textDataSetConfig()
    # seed: Optional[int] = II("common.seed")

@dataclass
class Kg2textCommonConfig(FairseqDataclass):

    logging_steps: int = field(default=20, metadata={"help": "default=20"})
    printing_steps: int = field(default=500, metadata={"help": "default=500"})
    save_every_n_epochs: int = field(default=10, metadata={"help": "default=10"})
    log_interval: int = field(default=100, metadata={"help": "default=100"})
    log_format: str = field(default="json", metadata={"help": "default: json"})
    tensorboard_logdir: str = field(default="", metadata={"help": "To implement"})
    seed: int = field(default=0, metadata={"help": "To implement"})
    cpu: bool = field(default=False, metadata={"help": "To implement"})
    gpu: bool = field(default=True, metadata={"help": "To implement"})
    tpu: bool = field(default=False, metadata={"help": "To implement"})
    epoches: int = field(default=10, metadata={"help": "To implement"})
    starting_epoch: int = field(default=0, metadata={"help": "To implement"})
    output_folder: str = field(default="", metadata={"help": "."})

@dataclass
class Kg2textConfig(FairseqDataclass):
    common: Kg2textCommonConfig = Kg2textCommonConfig()
    # for mbart50 transformer decoder:
    task: Kg2textTaskConfig = Kg2textTaskConfig()
    tokenizer: Kg2textTokenConfig = Kg2textTokenConfig()
    model: Kg2textModelConfig = Kg2textModelConfig()
    # tokenizers of encoder and decoder

from fairseq.dataclass.configs import *
from dataclasses import _MISSING_TYPE, dataclass, field
@dataclass
class FairseqConfig(FairseqDataclass):
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    bmuf: FairseqBMUFConfig = FairseqBMUFConfig()
    generation: GenerationConfig = GenerationConfig()
    eval_lm: EvalLMConfig = EvalLMConfig()
    interactive: InteractiveConfig = InteractiveConfig()
    model: Any = MISSING
    task: Any = None
    criterion: Any = None
    optimizer: Any = None
    lr_scheduler: Any = None
    scoring: Any = None
    bpe: Any = None
    tokenizer: Any = None