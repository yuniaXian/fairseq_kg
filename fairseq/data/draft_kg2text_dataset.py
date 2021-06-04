from fairseq.data import Dictionary
import torch
import numpy as np

class KgEmbedding:

    def __init__(self, dictionary) -> None:
        self.dict = dictionary
        self.ent = self.dict.index("[ENT]")
        self.triple = self.dict.index("[TRIPLE]")
        self.pred = self.dict.index("[PRED]")
        self.sub = self.dict.index("[SUB]")
        self.lang = self.dict.index("[en_XX]")
        self.eos = self.dict.eos()
        
    def get_property_embedding(self, source):
        # kg2text: [TRIPLE] [ENT] ... [PRED]...[SUB]...[TRIPLE]
        # sub: [SUB], [TRIPLE]
        # pred: [PRED], [SUB]
        # ent/obj: [ENT], [PRED]

        pass


    def get_tagging_intervals(self, tag_s, tag_t, source):
        
        
        """
        kg2text: [TRIPLE] [ENT] ... [PRED]...[SUB]...[TRIPLE]
        sub: [SUB], [TRIPLE]
        pred: [PRED], [SUB]
        ent/obj: [ENT], [PRED]
        
        kgpt: [ENT] ... [TRIPLE] [PRED] ... [SUB] ... [TRIPLE]
        sub: [SUB], [TRIPLE]
        pred: [PRED], [SUB]
        ent/obj: [ENT], [TRIPLE]
        entity_embedding: [ENT], [ENT]
        """
        # Note: make sure [tag_s, nearest_tag_t] is the section that you want for
        # tag_s (or tag_t) <--> (1 tot 1) wanted section 
        # it doesn't works for these cases:
        # [tag_s, tag_s, .., tag_s, tag_t, ..., tag_t]

        #source_lst = source.tolist()
        starting_indices = (source == tag_s).nonzero(as_tuple=True)[0]
        ending_indices = (source == tag_t).nonzero(as_tuple=True)[0]
        #ending_indices = torch.tensor([source_lst[starting_indices[i]+1:].index(tag_t) for i \
        #in range(starting_indices.size(0))], dtype=starting_indices.dtype)
        if starting_indices.size(0)>0 and ending_indices.size(0)>0:
            intervals_ind = (ending_indices > starting_indices.unsqueeze(1))
            nearest_ind = intervals_ind * torch.arange(intervals_ind.shape[1], 0, -1)
            ind = torch.argmax(nearest_ind, 1)
            ending_indices = ending_indices[ind]
            mask = starting_indices<ending_indices

            # [3,8,17] -> 0-2:1 3-8: 2 9-17:3  (17:last index)
            st_indices = torch.stack([starting_indices, ending_indices], 1)
            st_indices = st_indices[mask]
            return st_indices
        else:
            return torch.tensor([], dtype=source.dtype)

    def get_triple_embedding(self, source):
        # [TRIPLE] [ENT] ... [PRED]...[SUB]...[TRIPLE]
        # find triple token
        interval_indices = self.get_tagging_intervals(self.triple, self.triple)
        # [3,8,17] -> 0-2:1 3-8: 2 9-17:3  (17:last index)
        embedding = torch.zeros(source.size(0))
        #embedding[range(st_indices[], st_indices[])] = 1
        for i in range(interval_indices.size(0)):
            embedding[range(interval_indices[i][0], interval_indices[i][1])] = i+1

        #return embedding


    def get_one_triple_embedding_kgpt(self, source, embedding):
        ent_index = (source == self.ent).nonzero()
        interval_indices = self.get_tagging_intervals(self.pred, self.triple, source)

        embedding[range(ent_index, interval_indices[0][0]-1)] = 1
        embedding[interval_indices[0][0]-1] = 2


        for i in range(interval_indices.size(0)):
            embedding[range(interval_indices[i][0], interval_indices[i][1]+1)] = i+2
        #return embedding

    def get_triple_embedding_kgpt(self, source):
        # find triple token
        ent_indices = (source==self.ent).nonzero() # tenor([[1], [4], ...])
        intervals_ent_pred = self.get_tagging_intervals(self.ent, self.pred, source)
        intervals_pred_triple = self.get_tagging_intervals(self.pred, self.triple, source)
        #interval_ent_ent = self.get_tagging_intervals(self.ent, self.ent, source)
        # [3,8,17] -> 0-2:1 3-8: 2 9-17:3  (17:last index)
        embedding = torch.zeros(source.size(0))
        #embedding[range(st_indices[], st_indices[])] = 1

        for i in range(ent_indices.size(0)):
            s, t = ent_indices[i], ent_indices[i+1] if i+1<ent_indices.size(0) else embedding.size(0)
            self.get_one_triple_embedding_kgpt(source[s : t], embedding[s : t])
        return embedding

    
    def get_entity_embedding_kgpt(self, source):
        ent_indices = (source==self.ent).nonzero()
        #interval_indices = self.get_tagging_intervals(self.ent, self.ent, source)
        # [3,8,17] -> 0-2:1 3-8: 2 9-17:3  (17:last index)
        embedding = torch.zeros(source.size(0))
        #embedding[range(st_indices[], st_indices[])] = 1
        if ent_indices.size(0) == 1:
            embedding[ent_indices[0]:] = 1
        else:
            for i in range(ent_indices.size(0)-1):
                embedding[range(ent_indices[i], ent_indices[i+1])] = i+1

        return embedding


    def get_entity_embedding_kg2text(self, source):
        interval_indices = self.get_tagging_intervals(self.ent, self.pred)
        # [3,8,17] -> 0-2:1 3-8: 2 9-17:3  (17:last index)
        embedding = torch.zeros(source.size(0))
        #embedding[range(st_indices[], st_indices[])] = 1
        for i in range(interval_indices.size(0)):
            embedding[range(interval_indices[i][0], interval_indices[i][1]+1)] = i+1

        return embedding

    def get_kgpt_embedding(self, source):
        
        triple_embedding = self.get_triple_embedding_kgpt(source)
        entity_embedding = self.get_entity_embedding_kgpt(source)
        position_embedding = torch.arange(len(source)) # includes eos, bos

        return {
            "triple_ids": triple_embedding,
            "entity_ids": entity_embedding,
            "position_ids": position_embedding
        }

    def get_property_embedding(self, source):
        pass


if __name__ == "__main__":
    tgt_dict = Dictionary.load("/home/xianjiay/efs-storage/tokenizer/mbart50/dict/dict.mbart50_wtags.txt")
    emb = KgEmbedding(tgt_dict)
    triple = "[en_XX] [ENT] ▁Blues [TRIPLE] [PRED] ▁description [SUB] ▁Blues [TRIPLE] [PRED] ▁stylist ic ▁origin [SUB] ▁Rock ▁and ▁roll [TRIPLE] [ENT] ▁Blues [TRIPLE] [PRED] ▁description [SUB] ▁Blues [TRIPLE] [PRED] ▁stylist ic ▁origin [SUB] ▁Rock ▁and ▁roll [TRIPLE]"
    text = "[en_XX] ▁Rock ▁and ▁roll ▁music ▁origina ted ▁from ▁blues ▁music ▁ ."
    src_token = tgt_dict.encode_line(triple, add_if_not_exist=False, append_eos=True, reverse_order=False).long()
    tgt_token = tgt_dict.encode_line(text, add_if_not_exist=False, append_eos=True, reverse_order=False).long()
    src_sizes = []
    src_sizes.append(len(src_token))
    src_sizes = np.array(src_sizes)
    source = src_token

    res = emb.get_kgpt_embedding(source)
    res = emb.get_tagging_intervals(tgt_dict.index("[TRIPLE]"), tgt_dict.index("[TRIPLE]"), source)
    res = emb.get_triple_embedding(source)


    triple = "[en_XX] [ENT] ▁12 467 [TRIPLE] [PRED] ▁description [SUB] ▁12 467 [TRIPLE] [PRED] ▁1 st ▁run way ▁length ▁feet [SUB] ▁Ash ga bat ▁International ▁Airport [TRIPLE] [ENT] ▁211 [TRIPLE] [PRED] ▁description [SUB] ▁211 [TRIPLE] [PRED] ▁eleva tion ▁above ▁the ▁sea ▁level ▁( in ▁metres ) [SUB] ▁Ash ga bat ▁International ▁Airport [TRIPLE]"
    text = "[en_XX] ▁The ▁1 st ▁run way ▁at ▁Ash ga bat ▁International ▁airport ▁is ▁12 467 ▁feet ▁in ▁length ▁and ▁the ▁Airport ▁is ▁eleva ted ▁211 ▁metres ▁above ▁sea ▁level ▁ ."
    
