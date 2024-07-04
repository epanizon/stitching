import torch
from utils.comparison_utils import entropy_from_logits, correct_guess
from stitched_classes import bridgedLlamaFromModelToModel
import sys

def bridged_performance_eval(model_dict, 
                             model_dict2, 
                             model_name,
                             model_name2,
                             sizes_anchors, 
                             indexes_low, 
                             indexes_high, 
                             dataloader,
                             printout=False ):
    # ONLY PERFORMANCE from single layer to another layer
    Ndata = len(dataloader)
    text_results = []
    with torch.no_grad():
        for index_high in indexes_high:
            for index_low in indexes_low:
                # ATTENZIONE COERENZA DEGLI INDICI!!! CONTROLLARE
                representation_layercut_low_float = torch.load(f'./representations/representation_layercut_{index_low}_{model_name}.pt').float()
                representation_layercut_high_float = torch.load(f'./representations/representation_layercut_{index_high}_{model_name2}.pt').float()
                num_tot, _ = representation_layercut_low_float.shape
                for num_anchors in size_anchors:
                    for iter_over_size in range(3):
                        id_anchors = torch.randperm(num_tot)[:num_anchors] 
                        A = representation_layercut_low_float[id_anchors]
                        B = representation_layercut_high_float[id_anchors]
                        lstsq_fit_transform = torch.linalg.lstsq(A, B).solution
                        fl = bridgedLlamaFromModelToModel(model_dict, model_dict2, index_low, index_high, 32, transform=lstsq_fit_transform)
                        Ncorr = 5
                        correct_bridge = torch.zeros(Ncorr)
                        for ib, batch in enumerate(dataloader):
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            answer_id = batch['labels'][0].to(device)
                            final = fl.forward(input_ids, attention_mask)
                            p_model = torch.softmax(final[0,-1], 0)
                            ent = entropy_from_logits(final[0,-1])
                            correct_bridge += correct_guess(final[0,-1], answer_id, N=Ncorr)
                            del(input_ids, attention_mask, final, p_model)
                            torch.cuda.empty_cache()
                        text_result = f'{index_low} {index_high} {num_anchors}'+ str(*(correct_bridge/Ndata).data.cpu().float().numpy())
                        text_results.append(text_result)
                        if printout:
                            print(text_result)
                        del(fl, A, B, correct_bridge)
                        torch.cuda.empty_cache()
                del representation_layercut_low_float, representation_layercut_high_float
                torch.cuda.empty_cache()
    file.close()



def stitched_performance_eval(model, 
                              model_name,
                              nums_anchors, 
                              indexes_low, 
                              indexes_high, 
                              dataloader,
                              representation_folder='./representations',
                              printout=False ):
    Ndata = len(dataloader)
    if model_name in ["llama-2-7b", "llama-3-8b"]:
        index_max = 32
    elif model_name in ["llama-2-13b"]:
        index_max = 40
    else
        sys.stdout.flush()
        raise NameError(
            f"{model_name} not supported. Possible values are: 'llama-2-7b', 'llama-2-13b, 'llama-3-8b'"
        )
    
    text_results = []
    for index_high in indexes_high:
        representation_layercut_high_float = torch.load(representation_folder + f'./representations/representation_layercut_{index_high}_{model_name}.pt').float()
        for index_low in indexes_low:
            representation_layercut_low_float = torch.load(representation_folder + f'./representations/representation_layercut_{index_low}_{model_name}.pt').float()
            num_tot, _ = representation_layercut_low_float.shape
            for num_anchors in nums_anchors:
                for iter_over_size in range(1):
                    id_anchors = torch.randperm(num_tot)[:num_anchors] 
                    A = representation_layercut_low_float[id_anchors]
                    B = representation_layercut_high_float[id_anchors]
                    lstsq_fit_transform = torch.linalg.lstsq(A, B).solution
                    fl = stitchedLlamaFromModel(model, index_low, index_high, 32, transform=lstsq_fit_transform)
                    N = 1
                    success = 0.
                    global_overlap = 0.
                    for ib, batch in enumerate(dataloader):
                        input_ids = batch['input_ids'][:,:].to(device)
                        attention_mask = batch['attention_mask'][:,:].to(device)
                        # full llama8b model
                        final_real = model.forward(input_ids, output_hidden_states=False)
                        best_real = best_guesses(final_real['logits'][0,-1], tokenizer, N=N)
                        p_best = torch.softmax(final_real['logits'][0,-1], 0)
                        ent_real = entropy_from_logits(final_real['logits'][0,-1])
                        # now stitched model
                        final = fl.forward(input_ids, attention_mask)
                        p_model = torch.softmax(final[0,-1], 0)
                        best = best_guesses(final[0,-1], tokenizer, N=N)
                        ent = entropy_from_logits(final[0,-1])
                        over = overlap(p_best, p_model)
                        over_best = best_overlap(best_real, best, N)
                        success += over_best
                        global_overlap += over
                        del(input_ids, attention_mask, final, final_real, p_model, p_best)
                        torch.cuda.empty_cache()
                    text_result = f'{index_low} {index_high} {num_anchors} {iter_over_size} {success/len(dataloader)} {global_overlap/len(dataloader)} {ent} {ent_real}'
                    text_results.append(text_result)
                    if printout:
                        print(text_result)
                    del(fl, A, B, success, global_overlap, ent, ent_real)
                    torch.cuda.empty_cache()
            del representation_layercut_low_float
            torch.cuda.empty_cache()
        del representation_layercut_high_float
        torch.cuda.empty_cache()
    return text_results
