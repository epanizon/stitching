import torch
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
import os

device = f'cuda'

"""
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
"""

class stitchedLlamaFromModel():
    """
    stitchedLlamaFromModel 
    takes all layers of the model up to, and excluding, the Attention Layer #low_int.
    Therefore, it produces, up to the level of the cut, the equivalent of "representation_layercut_{low_int}.pt".
    Then, it uses all layers of the model from, and including, the Attention Layer #high_int.
    Therefore, it "assumes", after the cut, the equivalent of "representation_layercut_{high_int}.pt"
    """
    def __init__(self, model, low_int, high_int, max_int, transform=None):
        # Create inner transform
        _, self.emb_dim = model.get_submodule('model.embed_tokens').weight.shape
        if transform is not None:
            assert transform.shape[0] == transform.shape[1] == self.emb_dim, 'Stitching transform has wrong size'
            self.transform = transform.bfloat16()
        else:
            self.transform = torch.eye(self.emb_dim).bfloat16().to('cuda')
        # initial embedding
        self.layers_before = [model.get_submodule('model.embed_tokens')]
        # intermediate layers are used until cut
        for i in range(low_int):
            self.layers_before.append(model.get_submodule(f'model.layers.{i}'))
        # intermediate layers after cut
        self.layers_after = []
        for j in range(high_int, max_int):
            self.layers_after.append(model.get_submodule(f'model.layers.{j}'))
        # final norm + un-embedding
        self.layers_after.append(model.get_submodule('model.norm'))
        self.layers_after.append(model.get_submodule('lm_head'))
        # attention mask utility
        self.AMC = AttentionMaskConverter(True)
    
    @torch.no_grad()
    def forward(self, input_ids, attention_mask, all_x=False):
        input_ids = input_ids.to(device).view(1,-1)
        attention_mask = attention_mask.to(device).view(1,-1)
        _, seqlen = input_ids.shape
        causal_mask = self.AMC.to_causal_4d(1, seqlen, seqlen, torch.bfloat16).to('cuda')
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        x = input_ids
        xs = [x]
        # embedding
        x = self.layers_before[0].eval()(x)
        if all_x:
            xs.append(x)
        for l in self.layers_before[1:]:
            x = l.eval()(hidden_states=x, attention_mask=causal_mask, position_ids=position_ids)[0]
            if all_x:
                xs.append(x)
        # stitching transformation
        x = torch.matmul(x, self.transform)
        if all_x:
                    xs.append(x)
        # layers after stitching
        if len(self.layers_after)>2:
            for l in self.layers_after[:-2]:
                x = l.eval()(hidden_states=x, attention_mask=causal_mask, position_ids=position_ids)[0]
                if all_x:
                    xs.append(x)
        x = self.layers_after[-2].eval()(x)
        if all_x:
            xs.append(x)
        x = self.layers_after[-1].eval()(x)
        if all_x:
            xs.append(x)
            return x, xs
        return x

    
class bridgedLlamaFromModelToModel():
    def __init__(self, model1, model2, low_int, high_int, max_int, transform=None):
        # Create inner transform
        _, self.emb_dim = model1['model.embed_tokens'].weight.shape
        if transform is not None:
            assert transform.shape[0] == self.emb_dim, 'Stitching transform has wrong #1 size'
            #_, emb_dim2 = model2['model.embed_tokens'].weight.shape
            #print(transform.shape[1], emb_dim2)
            #assert transform.shape[1] == emb_dim2, 'Stitching transform has wrong #2 size '
            self.transform = transform.bfloat16()
        else:
            self.transform = torch.eye(self.emb_dim).bfloat16().to('cuda')
        # initial embedding
        self.layers_before = [model1['model.embed_tokens']]
        # intermediate layers are used until cut
        for i in range(low_int):
            self.layers_before.append(model1[f'model.layers.{i}'])
        # intermediate layers after cut
        self.layers_after = []
        for j in range(high_int, max_int):
            self.layers_after.append(model2[f'model.layers.{j}'])
        # final norm + un-embedding
        self.layers_after.append(model2['model.norm'])
        self.layers_after.append(model2['lm_head'])
        # attention mask utility
        self.AMC = AttentionMaskConverter(True)
    
    @torch.no_grad()
    def forward(self, input_ids, attention_mask, all_x=False):
        input_ids = input_ids.to(device).view(1,-1)
        attention_mask = attention_mask.to(device).view(1,-1)
        _, seqlen = input_ids.shape
        causal_mask = self.AMC.to_causal_4d(1, seqlen, seqlen, torch.bfloat16).to('cuda')
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        x = input_ids
        xs = [x]
        # embedding
        x = self.layers_before[0].eval()(x)
        if all_x:
            xs.append(x)
        for l in self.layers_before[1:]:
            x = l.eval()(hidden_states=x, attention_mask=causal_mask, position_ids=position_ids)[0]
            if all_x:
                xs.append(x)
        # stitching transformation
        x = torch.matmul(x, self.transform)
        if all_x:
                    xs.append(x)
        # layers after stitching
        if len(self.layers_after)>2:
            for l in self.layers_after[:-2]:
                x = l.eval()(hidden_states=x, attention_mask=causal_mask, position_ids=position_ids)[0]
                if all_x:
                    xs.append(x)
        x = self.layers_after[-2].eval()(x)
        if all_x:
            xs.append(x)
        x = self.layers_after[-1].eval()(x)
        if all_x:
            xs.append(x)
            return x, xs
        return x

class bridgedLlamaFromModelToModel_devControl():
    def __init__(self, 
                 model1, 
                 model2, 
                 low_int, 
                 high_int, 
                 max_int, 
                 transform=None, 
                 model1_device='cuda:0',
                 model2_device='cuda:1'):
        # Create inner transform
        _, self.emb_dim = model1['model.embed_tokens'].weight.shape
        if transform is not None:
            assert transform.shape[0] == self.emb_dim, 'Stitching transform has wrong #1 size'
            #_, emb_dim2 = model2['model.embed_tokens'].weight.shape
            #print(transform.shape[1], emb_dim2)
            #assert transform.shape[1] == emb_dim2, 'Stitching transform has wrong #2 size '
            self.transform = transform.bfloat16().to(model1_device)
        else:
            self.transform = torch.eye(self.emb_dim).bfloat16().to(model1_device)
        # initial embedding
        self.layers_before = [model1['model.embed_tokens']]
        # intermediate layers are used until cut
        for i in range(low_int):
            self.layers_before.append(model1[f'model.layers.{i}'])
        # intermediate layers after cut
        self.layers_after = []
        for j in range(high_int, max_int):
            self.layers_after.append(model2[f'model.layers.{j}'])
        # final norm + un-embedding
        self.layers_after.append(model2['model.norm'])
        self.layers_after.append(model2['lm_head'])
        # attention mask utility
        self.AMC = AttentionMaskConverter(True)
        self.dev1=model1_device
        self.dev2=model2_device
    
    @torch.no_grad()
    def forward(self, input_ids, attention_mask, all_x=False):
        dev1 = self.dev1
        dev2 = self.dev2
        input_ids = input_ids.to(dev1).view(1,-1)
        attention_mask = attention_mask.to(dev1).view(1,-1)
        _, seqlen = input_ids.shape
        causal_mask = self.AMC.to_causal_4d(1, seqlen, seqlen, torch.bfloat16).to(dev1)
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        x = input_ids
        xs = [x]
        # embedding
        x = self.layers_before[0].eval()(x)
        if all_x:
            xs.append(x)
        for l in self.layers_before[1:]:
            x = l.eval()(hidden_states=x, attention_mask=causal_mask, position_ids=position_ids)[0]
            if all_x:
                xs.append(x)
        # stitching transformation
        x = torch.matmul(x, self.transform)
        if all_x:
                    xs.append(x)
        # layers after stitching
        x.to(dev2)
        if len(self.layers_after)>2:
            for l in self.layers_after[:-2]:
                x = l.eval()(hidden_states=x, attention_mask=causal_mask, position_ids=position_ids)[0]
                if all_x:
                    xs.append(x)
        x = self.layers_after[-2].eval()(x)
        if all_x:
            xs.append(x)
        x = self.layers_after[-1].eval()(x)
        if all_x:
            xs.append(x)
            return x, xs
        return x
