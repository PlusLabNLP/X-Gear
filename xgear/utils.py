import numpy as np
from tensorboardX import SummaryWriter

class Summarizer(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

def generate_vocabs(datasets):
    event_type_set = set()
    role_type_set = set()
    for dataset in datasets:
        event_type_set.update(dataset.event_type_set)
        role_type_set.update(dataset.role_type_set)
    
    event_type_itos = sorted(event_type_set)
    role_type_itos = sorted(role_type_set)
    
    event_type_stoi = {k: i for i, k in enumerate(event_type_itos)}
    role_type_stoi = {k: i for i, k in enumerate(role_type_itos)}
    
    return {
        'event_type_itos': event_type_itos,
        'event_type_stoi': event_type_stoi,
        'role_type_itos': role_type_itos,
        'role_type_stoi': role_type_stoi,
    }

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

def cal_scores(gold_roles, pred_roles):
    # argument identification
    gold_arg_id_num, pred_arg_id_num, match_arg_id_num = 0, 0, 0
    for gold_role, pred_role in zip(gold_roles, pred_roles):
        gold_set = set([(r[0][2],)+r[1][:-1] for r in gold_role])
        pred_set = set([(r[0][2],)+r[1][:-1] for r in pred_role])
        
        gold_arg_id_num += len(gold_set)
        pred_arg_id_num += len(pred_set)
        match_arg_id_num += len(gold_set & pred_set)
        
    # argument classification
    gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num = 0, 0, 0
    for gold_role, pred_role in zip(gold_roles, pred_roles):
        gold_set = set([(r[0][2],)+r[1] for r in gold_role])
        pred_set = set([(r[0][2],)+r[1] for r in pred_role])
        
        gold_arg_cls_num += len(gold_set)
        pred_arg_cls_num += len(pred_set)
        match_arg_cls_num += len(gold_set & pred_set)
    
    scores = {
        'arg_id': (gold_arg_id_num, pred_arg_id_num, match_arg_id_num) + compute_f1(pred_arg_id_num, gold_arg_id_num, match_arg_id_num),
        'arg_cls': (gold_arg_cls_num, pred_arg_cls_num, match_arg_cls_num) + compute_f1(pred_arg_cls_num, gold_arg_cls_num, match_arg_cls_num),
    }
    
    return scores

def get_span_idxs(pieces, token_start_idxs, span, tokenizer, trigger_span=None):
    """
    Convert the predicted span to the offsets in the passage.
    Return (-1, -1) if there is no matched string.
    """
    
    t_span = span.split(' ')
    
    words = []
    for s in t_span:
        words.extend(tokenizer.encode(s, add_special_tokens=True)[:-1]) # ignore eos
    
    candidates = []
    for i in range(len(pieces)):
        j = 0
        k = 0
        while j < len(words) and i+k < len(pieces):
            if pieces[i+k] == words[j]:
                j += 1
                k += 1
            elif tokenizer.decode(words[j]) == "":
                j += 1
            elif tokenizer.decode(pieces[i+k]) == "":
                k += 1
            else:
                break
        if j == len(words):
            candidates.append((i, i+k))
            
    candidates = [(token_start_idxs.index(c1), token_start_idxs.index(c2))for c1, c2 in candidates if c1 in token_start_idxs and c2 in token_start_idxs]
    if len(candidates) < 1:
        return -1, -1
    else:
        if trigger_span is None:
            return candidates[0]
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))[0]
        
def get_span_idxs_zh(tokens, span, trigger_span=None):
    """
    Convert the predicted span to the offsets in the passage for Chinese.
    Return (-1, -1) if there is no matched string.
    """
    
    candidates = []
    for i in range(len(tokens)):
        for j in range(i, len(tokens)):
            c_string = "".join(tokens[i:j+1])
            if c_string == span:
                candidates.append((i, j+1))
                break
            elif not span.startswith(c_string):
                break
                
    if len(candidates) < 1:
        return -1, -1
    else:
        if trigger_span is None:
            return candidates[0]
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))[0]