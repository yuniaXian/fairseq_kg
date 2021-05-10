import argparse
import fairseq
from fairseq.data import Dictionary
from torch import nn
import torch
import os

def load_model(folder):
    path = f'{folder}/model.pt'
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [path],
        arg_overrides = {
            'lang_dict': f'{folder}/ML50_langs.txt',
            'data': folder
        }
    )
    return model[0], model[0].decoder.dictionary, cfg

def load_state(folder):
    return fairseq.checkpoint_utils.load_checkpoint_to_cpu(
        f'{folder}/model.pt',
        arg_overrides = {
            'lang_dict': f'{folder}/ML50_langs.txt',
            'data': folder
        }        
    )


def enlarge_weights(weights, loaded_dict_size, new_embeds_to_add):
    loaded_mask_token_embedding = weights[-1, :]
    weights = torch.cat(
        [
            weights[
                : loaded_dict_size - 1, :
            ],
            new_embeds_to_add,
            # mask token needs to be at the end
            loaded_mask_token_embedding.unsqueeze(0),
        ]
    )
    return weights

def augment_mbart_state_embeddings(state, tgt_dict):
    state_dict = state['model']
    loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
    num_embeddings_toadd = len(tgt_dict) - loaded_dict_size
    if num_embeddings_toadd == 0:
        print('the dictionary is the same, no need to augment')
        return state
    elif num_embeddings_toadd < 0:
        raise ValueError("target dict is smaller than current dict")
        
    embed_dim = state_dict["encoder.embed_tokens.weight"].size(1)
    new_embeds_to_add = torch.zeros(
        num_embeddings_toadd, embed_dim, dtype=state_dict["decoder.embed_tokens.weight"].dtype)
    nn.init.normal_(new_embeds_to_add, mean=0, std=embed_dim ** -0.5)
    
    state_dict["encoder.embed_tokens.weight"] = enlarge_weights(
        state_dict["encoder.embed_tokens.weight"], loaded_dict_size, new_embeds_to_add)
    state_dict["decoder.embed_tokens.weight"] = enlarge_weights(
        state_dict["decoder.embed_tokens.weight"], loaded_dict_size, new_embeds_to_add)
#     state_dict["decoder.output_projection.weight"] = enlarge_weights(
#         state_dict["decoder.output_projection.weight"], loaded_dict_size, new_embeds_to_add)  
    return state

if __name__ == "__main__":
    # usage example:
    # python /efs-storage/workspaces/hoverboard/MultiTaskWave/src/MultiTaskWave/scripts/model_utils/augment_mbart_model.py /efs-storage/models/mbart50/mbart50.ft.nn --tgt-dict /efs-storage/workspaces/hoverboard/MultiTaskWave/src/MultiTaskWave/resources/mbart50/dict.mbart50_extra_labels.txt --save-to /efs-storage/models/mbart50/mbart50.ft.nn.extra_syms
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument('--save-to', type=str)
    parser.add_argument('--tgt-dict', type=str)

    args = parser.parse_args()

    tgt_dict = Dictionary.load(args.tgt_dict)
    state = load_state(args.folder)
    new_state = augment_mbart_state_embeddings(state, tgt_dict)

    new_folder = args.save_to
    os.makedirs(new_folder, exist_ok=True)
    new_model_path = f"{new_folder}/model.pt"
    torch.save(new_state, f=open(new_model_path, 'wb'))
    print('new model save to: ', new_model_path)