import torch
import math
import os, sys, time
import nltk
import sys
import json
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Model import TransformerDecoder, GatedTransformerDecoder, GraphGatedTransformerDecoder
from DataLoader import *
from transformers import GPT2Tokenizer
from types import SimpleNamespace
from torch.autograd import Variable
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from nltk.tokenize import word_tokenize
from omegaconf import OmegaConf
from myutils import *
from kg2text_model import *
from kg2text_task import *

def setup_tags(data_cfg, tokenizer):

        # TODO check if it also works for mbart tokenizer
        banwords = [tokenizer.encode(word) for word in ['It', 'She', 'They', 'He', 'it', 'she', 'he', 'they']]
        #banwords = [w[0] for w in banwords]
        banwords = [x.tolist()[0] if isinstance(x, Tensor) else x[0] for x in banwords ]

        if "mbart" in data_cfg.tokenizer_tgt or "sentencepiece" in data_cfg.tokenizer_tgt:
            eos = "</s>"
            bos = "<s>"
            pad = "<pad>"
            unk = "<unk>"
        elif "gpt2" in data_cfg.tokenizer_tgt:
            eos = "[EOS]"
            bos = "[SOS]"
            pad = "[PAD]"
            unk = "[UNK]"
        else:
            raise NotImplementedError
        
        if data_cfg.prepend_tgt_lang_tag:
            lang_label = data_cfg.tgt_lang_tag_template.format(data_cfg.tgt_lang)
            if data_cfg.tokenizer_tgt in ["sentencepiece", "mbart", "mbart50"]:
                prepend_label = task.tgt_dict.index(lang_label)
            else:
                prepend_label = tokenizer.encode(lang_label)[0]
        else:
            if data_cfg.tokenizer_tgt in ["sentencepiece", "mbart", "mbart50"]:
                prepend_label = task.tgt_dict.index(bos)
            else:
                prepend_label = tokenizer.encode(bos)[0] # TODO: check GPT2 tokenizer returns [50262]
        return eos, bos, pad, unk, prepend_label, banwords

def setup_random_seed(seed_val):
    seed_val = 0
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    now = time.strftime("%Y_%m%d_%H%M", time.localtime(time.time()))
    # set cfg, load config and update cfg
    args = get_my_args()
    print(args)
    cfg_new = Kg2textConfig()
    cfg = set_cfg(cfg_new, args, now)

    # random seed set
    setup_random_seed(args.seed)
    # logger setup

    task_cfg, data_cfg, model_cfg, common, token_cfg = cfg.task, getattr(cfg.task, args.data), cfg.model,cfg.common, cfg.tokenizer
    data_cfg.option, data_cfg.split = task_cfg.option, task_cfg.split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common.device = device.type

    record_file = "{encoder}_{decoder}_{task}_{data}_{time}_result.txt".format(
        encoder=model_cfg.encoder_type, decoder=model_cfg.decoder_type, task=cfg.task.option, data=cfg.task.dataset, time=now)

    logger.info("data: %s, task: %s, encoder: %s, decoder: %s, output_dir: %s",
                cfg.task.dataset, cfg.task.option, model_cfg.encoder_type, model_cfg.decoder_type,
                common.output_dir)
    logger.info("device: %s", common.device)

    # TODO setup bpe and dict in task and then to tokenizer
    tokenizer = get_tokenizer(data_cfg, "tgt")
    task = Kg2textTask.setup_task(cfg, args)
    eos, bos, pad, unk, prepend_label, banwords = setup_tags(data_cfg, tokenizer)

    # experiment entry point
    # experiment(task_cfg, data_cfg, model_cfg, common, token_cfg, task.source_dictionary, task.target_dictionary)

    data = task.load_dataset(data_cfg.split, False, data_cfg)
    model = Kg2textTransformerModel.build_model(model_cfg, task)
    model = torch.nn.DataParallel(model)
    model.to(device)

    # set random seed for dataloader
    setup_random_seed(0)
    
    if task_cfg.option in ['train', "finetune"]:
        
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=data_cfg.batch_size,
                                num_workers=data_cfg.num_workers)

        t_total = len(data) * common.epochs
        warmup_steps = int(t_total * 0.2)
        logger.info("Warming up for {} steps, and then start linearly decresing learning rate".format(warmup_steps))
        optimizer = optim.Adam(model.parameters(), common.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        loss_func.to(device)
        model.train()
        
        logging_loss = 0
        tr_loss = 0
        global_step = common.starting_epoch * len(dataloader)
        
        tb_writer = SummaryWriter(log_dir=common.output_dir)
        for epoch in trange(common.starting_epoch, common.epochs, desc="Epoch"):
            for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):

                ####################### train_step #################################################
                data = tuple(Variable(t).to(device) for t in batch)
                target = data[-1]    # data :(input_ids, enti_ids,..ids, ids, output_ids[:-1], output_ids[1:] )
                if global_step == 0:
                    ls_of_sz = [ x.size() for x in batch]
                    print(ls_of_sz)

                model.zero_grad()
                optimizer.zero_grad() # TODO ??
                #logits = model(src_tokens=data[:-2], src_lengths=data_cfg.src_length, prev_output_tokens=data[-2])

                # logtis: B x T(-1) x V, target: B x T (-1)
                # logits -> (B x T) x V,  target -> (B x T)
                if model_cfg.use_copy_gate:
                    logits = model.module.forward_v2(src_tokens=data[:-2], src_lengths=data_cfg.src_length,
                                                     prev_output_tokens=data[-2])
                else:
                    raise NotImplementedError
                loss = loss_func(logits.view(-1, logits.shape[-1]), target.view(-1))
                if math.isnan(loss): # Check whether a value is NaN or not:
                    import pdb
                    pdb.set_trace()

                tr_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

                global_step += 1

                if step > 0 and global_step % common.logging_steps == 0:
                    avg_loss = (tr_loss - logging_loss) / common.logging_steps
                    tb_writer.add_scalar("loss", avg_loss, global_step)
                    logging_loss = tr_loss

                    loss_log_file = common.loss_log_file+"_loss.log"
                    with open(loss_log_file, "a") as f:
                        f.write("{} {} \n".format(global_step, avg_loss))

            if epoch % common.save_every_n_epochs == 0 and epoch > 0:
                torch.save(model.module.state_dict(), '{}/model_ep{}.pt'.format(common.output_dir, epoch))
        torch.save(model.module.state_dict(), '{}/model_ep{}.pt'.format(common.output_dir, epoch))        
        tb_writer.close()

    elif task_cfg.option in ['eval', 'test']:

        eval_data = data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=data_cfg.batch_size,
                                     num_workers=data_cfg.num_workers)

        model.eval()

        results = []
        list_of_references = []
        list_of_hypothesis = []
        prepend_label = tokenizer.bos_token_id

        for step, batch in enumerate(eval_dataloader):
            data = tuple(Variable(t).to(common.device) for t in batch)
            #print("first data sample:", data[:1])
            if common.beam_size == 1:
                result = model.module.greedy_decode(data[:-2], src_lengths=data_cfg.src_length, prepend_label=prepend_label, tgt_lengths=data_cfg.tgt_length)
                #logits, extra = model.forward_v2(src_tokens=data[:-2], src_lengths=data_cfg.src_length,
                #                       prev_output_tokens=data[-2])
            else:
                result = model.module.beam_search(data[:-2], tokenizer, n_bm=common.beam_size, banwords=banwords, max_token_seq_len=data_cfg.tgt_length)
            for offset, r in enumerate(result):
                sent = tokenizer.decode(r, clean_up_tokenization_spaces=True)
                if eos in sent:
                    sent = sent[:sent.index(eos)].strip()

                idx = step * data_cfg.batch_size + offset
                references = eval_data.get_reference(idx)
                #print("references:", references[:1])
                tok_sent = word_tokenize(sent)
                #print(tok_sent)
                results.append(sent)
                list_of_hypothesis.append(tok_sent)
                list_of_references.append(references)

            if task_cfg.option == 'eval':
                bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypothesis)
                sys.stdout.write('finished {}/{}; BLEU{} {} \n'.format(step, len(eval_dataloader), common.bleu, bleu))
            else:
                sys.stdout.write('finished {}/{} \n'.format(step, len(eval_dataloader)))


        if task_cfg.option == 'eval':
            bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypothesis)
            print('finished {}/{}; BLEU{} {}'.format(step, len(eval_dataloader), common.bleu, bleu))
        with open(record_file, 'a') as f:
            for x in results:
                f.write(x + '\n')
            f.write("{}".format(bleu))

    else:
        raise NotImplementedError("This option is not yet supported")
