# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.data import encoders


def get_whole_word_mask(args, dictionary):
    bpe = encoders.build_bpe(args)
    if bpe is not None:


        def is_beginning_of_word(i):
            if i < dictionary.nspecial:
                # special elements are always considered beginnings
                return True
            tok = dictionary[i]
            if tok.startswith("madeupword"):
                return True
            try:
                return bpe.is_beginning_of_word(tok)
            except ValueError:
                return True

        mask_whole_words = torch.ByteTensor(
            list(map(is_beginning_of_word, range(len(dictionary))))
        )
        return mask_whole_words
    return None

def get_whole_word_mask1(args, dictionary):
    bpe = encoders.build_bpe(args)
    if bpe is not None:

        def is_beginning_of_word(i):
            if i < dictionary.nspecial or i > dictionary.index("[sl_SI]"):
                # special elements are always considered beginnings
                return True
            tok = dictionary[i]
            if tok.startswith("madeupword"):
                return True
            try:
                return bpe.is_beginning_of_word(tok)
            except ValueError:
                return True

        mask_whole_words = torch.ByteTensor(
            list(map(is_beginning_of_word, range(len(dictionary))))
        )
        return mask_whole_words
    return None

def get_special_token_mask(args, dictionary):
    if getattr(args, "extra_special_tokens", None) == None:
        extra_special_tokens = set(["[KG]", "[TEXT]", "[ENT]", "[TRIPLE]", "[PRED]", "[SUB]", "[SEP]", "<mask>", \
                '[ar_AR]','[cs_CZ]', '[de_DE]', '[en_XX]', '[es_XX]', \
                '[et_EE]', '[fi_FI]', '[fr_XX]', '[gu_IN]', '[hi_IN]', '[it_IT]', 
                '[ja_XX]', '[kk_KZ]', 
                '[ko_KR]', '[lt_LT]', '[lv_LV]', '[my_MM]', '[ne_NP]', '[nl_XX]', 
                '[ro_RO]', '[ru_RU]', '[si_LK]', '[tr_TR]', '[vi_VN]', '[zh_CN]', 
                '[af_ZA]', '[az_AZ]', '[bn_IN]', '[fa_IR]', '[he_IL]', '[hr_HR]', 
                '[id_ID]', '[ka_GE]', '[km_KH]', '[mk_MK]', '[ml_IN]', '[mn_MN]', 
                '[mr_IN]', '[pl_PL]', '[ps_AF]', '[pt_XX]', '[sv_SE]', '[sw_KE]', 
                '[ta_IN]', '[te_IN]', '[th_TH]', '[tl_XX]', '[uk_UA]', '[ur_PK]', 
                '[xh_ZA]', '[gl_ES]', '[sl_SI]'])
    else:
        extra_special_tokens = set(args.extra_special_tokens)
    
    def is_special_token(i):
        if i < dictionary.nspecial:
            # special elements are always considered beginnings
            return True
        tok = dictionary[i]
        if tok in extra_special_tokens:
            return True
        else:
            return False

    mask_speical_tokens = torch.ByteTensor(
        list(map(is_special_token, range(len(dictionary))))
    )
    return mask_speical_tokens