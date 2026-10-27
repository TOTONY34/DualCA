import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats

def sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type):
    """
    Computes the similarity matrix using pooled video frames
    Output: sims: num_texts x num_vids
    """
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    if vid_embeds_pooled.dim() == 2:
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())
    elif pooling_type == 'avg':
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())
    else:
        vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)
        text_embeds = text_embeds.unsqueeze(1)
        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)

    return sims


def sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, pooling_type):
    """
    Computes the similarity matrix for inference.
    """
    text_embeds_per_video_id = text_embeds_per_video_id / text_embeds_per_video_id.norm(dim=-1, keepdim=True)
    vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id / vid_embeds_pooled_per_video_id.norm(dim=-1, keepdim=True)

    if vid_embeds_pooled_per_video_id.dim() == 2:
        N_v, Max_T, D = text_embeds_per_video_id.shape
        flat_text = text_embeds_per_video_id.view(-1, D)
        sims_2d = torch.mm(flat_text, vid_embeds_pooled_per_video_id.t())
        sims = sims_2d.view(N_v, Max_T, N_v)
        return sims

    if pooling_type == 'avg':
        sims = text_embeds_per_video_id @ vid_embeds_pooled_per_video_id.t()
    else:
        num_vids, max_text_per_vid, embed_dim = text_embeds_per_video_id.shape
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.permute(1,2,3,0)
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.view(num_vids*max_text_per_vid, embed_dim, num_vids)
        text_embeds_per_video_id = text_embeds_per_video_id.unsqueeze(2)
        text_embeds_per_video_id = text_embeds_per_video_id.view(num_vids*max_text_per_vid, 1, embed_dim)
        sims = torch.bmm(text_embeds_per_video_id, vid_embeds_pooled_per_video_id)
        sims = sims.view(num_vids, max_text_per_vid, 1, num_vids).squeeze(2)
        
    return sims


def generate_embeds_per_video_id(text_embeds, vid_embeds_pooled, all_vid_ids, pooling_type):
    # Construct dictionary of text embeds per unique video id
    text_embeds_per_video_id = {}
    for idx, v_id in enumerate(all_vid_ids):
        if v_id in text_embeds_per_video_id:
            text_embeds_per_video_id[v_id].append(text_embeds[idx])
        else:
            text_embeds_per_video_id[v_id] = [text_embeds[idx]]

    for v_id in text_embeds_per_video_id:
        text_embeds_per_video_id[v_id] = torch.stack(text_embeds_per_video_id[v_id])

    text_embeds_per_video_id = pad_and_stack_dict_to_tensor(text_embeds_per_video_id,
        text_embeds_per_video_id.keys(), text_embeds.shape[-1])

    if vid_embeds_pooled.dim() == 2:
        unique_vid_map = {}
        for idx, v_id in enumerate(all_vid_ids):
            if v_id not in unique_vid_map:
                unique_vid_map[v_id] = vid_embeds_pooled[idx]
        
        keys_order = list(text_embeds_per_video_id.keys()) if hasattr(text_embeds_per_video_id, 'keys') else unique_vid_map.keys()
        
        
        vid_embeds_pooled_per_video_id = torch.stack([unique_vid_map[k] for k in text_embeds_per_video_id.keys() if k in unique_vid_map])
        
        return text_embeds_per_video_id, vid_embeds_pooled_per_video_id

    if pooling_type == 'avg':
        vid_embeds_pooled_per_video_id = vid_embeds_pooled
    else:
        vid_embeds_pooled_per_video_id = []
        for i in range(vid_embeds_pooled.shape[0]):
            vid_embeds_pooled_per_video_id.append({})
            for idx, v_id in enumerate(all_vid_ids):
                if v_id in vid_embeds_pooled_per_video_id[i]:
                    vid_embeds_pooled_per_video_id[i][v_id].append(vid_embeds_pooled[i, idx, :])
                else:
                    vid_embeds_pooled_per_video_id[i][v_id] = [vid_embeds_pooled[i, idx, :]]

        for i in range(len(vid_embeds_pooled_per_video_id)):
            for v_id in vid_embeds_pooled_per_video_id[i]:
                vid_embeds_pooled_per_video_id[i][v_id] = torch.stack(vid_embeds_pooled_per_video_id[i][v_id])
            vid_embeds_pooled_per_video_id[i] = pad_and_stack_dict_to_tensor(vid_embeds_pooled_per_video_id[i],
                    vid_embeds_pooled_per_video_id[i].keys(), vid_embeds_pooled.shape[-1])

        vid_embeds_pooled_per_video_id = torch.stack(vid_embeds_pooled_per_video_id)

    return text_embeds_per_video_id, vid_embeds_pooled_per_video_id


def t2v_metrics(sims):
    if sims.dim() == 2:
        if isinstance(sims, torch.Tensor):
            sims = sims.cpu().numpy()
        
        num_queries = sims.shape[0]
        ranks = np.zeros(num_queries)
        
        for index in range(num_queries):
            inds = np.argsort(sims[index])[::-1]
            
            match = np.where(inds == index)[0]
            if len(match) > 0:
                ranks[index] = match[0]
            else:
                ranks[index] = 99999 

        return compute_metrics(ranks)

    stacked_sims = sims.permute(1,0,2)
    sims_sort = torch.argsort(stacked_sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)
    ranks = torch.flatten(torch.diagonal(sims_sort_2, dim1=1, dim2=2))
    
    valid_check = torch.flatten(torch.diagonal(sims, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(valid_check), torch.isnan(valid_check))
    valid_ranks = ranks[mask]

    return compute_metrics(valid_ranks.numpy())


def v2t_metrics(sims):
    if sims.dim() == 2:
        return t2v_metrics(sims.t())

    sims[sims!=sims] = float('-inf')
    sims, _ = torch.max(sims, dim = 1)
    sims = sims.t()
    sims_sort = torch.argsort(sims, dim=-1, descending=True)
    sims_sort_2 = torch.argsort(sims_sort, dim=-1, descending=False)
    ranks = torch.diag(sims_sort_2).numpy()

    return compute_metrics(ranks)


def compute_metrics(lst):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(lst == 0)) / len(lst)
    metrics["R5"] = 100 * float(np.sum(lst < 5)) / len(lst)
    metrics["R10"] = 100 * float(np.sum(lst < 10)) / len(lst)
    metrics["R50"] = 100 * float(np.sum(lst < 50)) / len(lst)
    metrics["R100"] = 100 * float(np.sum(lst < 100)) / len(lst)
    metrics["MedR"] = np.median(lst) + 1
    metrics["MeanR"] = np.mean(lst) + 1
    return metrics


def pad_and_stack_dict_to_tensor(input, order, d=512):
    max_length = max([input[k].shape[0] for k in input])
    padded_input = {k: torch.cat([input[k], torch.full((max_length - input[k].shape[0], d), 
                                                float("-inf"), device = input[k].device)]) for k in input}
    padded_stacked_input = torch.stack([padded_input[k] for k in order], dim = 0)
    return padded_stacked_input