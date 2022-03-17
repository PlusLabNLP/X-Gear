import os, sys, json, logging, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import MT5Tokenizer
from model import GenerativeModel, Prefix_fn_cls
from data import EEDataset
from utils import cal_scores, get_span_idxs, get_span_idxs_zh
from argparse import ArgumentParser, Namespace

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True)
parser.add_argument('-m', '--model', required=True)
parser.add_argument('-o', '--output_dir', type=str, required=True)
parser.add_argument('--constrained_decode', default=False, action='store_true')
parser.add_argument('--beam', type=int, default=4)
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config = Namespace(**config)

# over write beam size
config.beam_size = args.beam

# import template file
if config.dataset == "ace05":
    from template_generate_ace import event_template_generator, IN_SEP, ROLE_LIST, NO_ROLE, AND
    TEMP_FILE = "template_generate_ace"
elif config.dataset == "ere":
    from template_generate_ere import event_template_generator, IN_SEP, ROLE_LIST, NO_ROLE, AND
    TEMP_FILE = "template_generate_ere"
else:
    raise NotImplementedError

# fix random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# set GPU device
torch.cuda.set_device(config.gpu_device)

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
        
# check valid styles
assert np.all([style in ['triggerword', 'template'] for style in config.input_style])
assert np.all([style in ['argument:roletype'] for style in config.output_style])
    
# tokenizer
if config.model_name.startswith("google/mt5-"):
    tokenizer = MT5Tokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
elif config.model_name.startswith("copy+google/mt5-"):
    model_name = config.model_name.split('copy+', 1)[1]
    tokenizer = MT5Tokenizer.from_pretrained(model_name, cache_dir=config.cache_dir)
else:
    raise NotImplementedError

special_tokens = []
sep_tokens = []
if "triggerword" in config.input_style:
    sep_tokens += [IN_SEP["triggerword"]]
if "template" in config.input_style:
    sep_tokens += [IN_SEP["template"]]
if "argument:roletype" in config.output_style:
    special_tokens += [f"<--{r}-->" for r in ROLE_LIST]
    special_tokens += [f"</--{r}-->" for r in ROLE_LIST]
    special_tokens += [NO_ROLE, AND]
tokenizer.add_tokens(sep_tokens+special_tokens)    

# load data
dev_set = EEDataset(tokenizer, config.dev_file, max_length=config.max_length)
test_set = EEDataset(tokenizer, config.test_file, max_length=config.max_length)
dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)
with open(config.vocab_file) as f:
    vocab = json.load(f)

# load model
logger.info(f"Loading model from {args.model}")
model = GenerativeModel(config, tokenizer)
model.load_state_dict(torch.load(args.model, map_location=f'cuda:{config.gpu_device}'))
model.cuda(device=config.gpu_device)
model.eval()

# output directory
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# eval dev set
progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev')
dev_gold_triggers, dev_gold_roles, dev_pred_roles = [], [], []
dev_pred_wnd_ids, dev_gold_outputs, dev_pred_outputs, dev_inputs = [], [], [], []
for batch in DataLoader(dev_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=dev_set.collate_fn):
    progress.update(1)

    batch_pred_roles = [[] for _ in range(config.eval_batch_size)]
    batch_pred_outputs = [[] for _ in range(config.eval_batch_size)]
    batch_gold_outputs = [[] for _ in range(config.eval_batch_size)]
    batch_inputs = [[] for _ in range(config.eval_batch_size)]
    batch_event_templates = []
    for tokens, triggers, roles in zip(batch.tokens, batch.triggers, batch.roles):
        batch_event_templates.append(event_template_generator(tokens, triggers, roles, config.input_style, config.output_style, vocab, config.lang))
    
    # convert EE instances to EAE instances 
    eae_inputs, eae_gold_outputs, eae_events, eae_bids = [], [], [], []
    for i, event_temp in enumerate(batch_event_templates):
        for data in event_temp.get_training_data():
            eae_inputs.append(data[0])
            eae_gold_outputs.append(data[1])
            eae_events.append(data[2])
            eae_bids.append(i)
            batch_inputs[i].append(data[0])
    
    # if there are triggers in this batch, predict argument roles
    if len(eae_inputs) > 0:
        eae_inputs = tokenizer(eae_inputs, return_tensors='pt', padding=True, max_length=config.max_length+2)
        enc_idxs = eae_inputs['input_ids']
        enc_idxs = enc_idxs.cuda()
        enc_attn = eae_inputs['attention_mask'].cuda()

        if config.beam_size == 1:
            model.model._cache_input_ids = enc_idxs
        else:
            expanded_return_idx = (
                torch.arange(enc_idxs.shape[0]).view(-1, 1).repeat(1, config.beam_size).view(-1).to(enc_idxs.device)
            )
            input_ids = enc_idxs.index_select(0, expanded_return_idx)
            model.model._cache_input_ids = input_ids
        
        # inference
        with torch.no_grad():
            if args.constrained_decode:
                prefix_fn_obj = Prefix_fn_cls(tokenizer, ["[and]"], enc_idxs)
                outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                        num_beams=config.beam_size, 
                        max_length=config.max_output_length,
                        forced_bos_token_id=None,
                        prefix_allowed_tokens_fn=lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
                        )
            else:
                outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                    num_beams=config.beam_size, max_length=config.max_output_length, 
                    forced_bos_token_id=None)
        eae_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
        
        # extract argument roles from the generated outputs
        for p_text, g_text, info, bid in zip(eae_pred_outputs, eae_gold_outputs, eae_events, eae_bids):
            theclass = getattr(sys.modules[TEMP_FILE], info['event type'].replace(':', '_').replace('-', '_'), False)
            assert theclass
            template = theclass(config.input_style, config.output_style, info['tokens'], info['event type'], config.lang, info)

            pred_object = template.decode(p_text)

            for span, role_type, _ in pred_object:
                # convert the predicted span to the offsets in the passage
                # Chinese uses a different function since there is no space between Chenise characters
                if config.lang == "chinese":
                    sid, eid = get_span_idxs_zh(batch.tokens[bid], span, trigger_span=info['trigger span'])
                else:
                    sid, eid = get_span_idxs(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer, trigger_span=info['trigger span'])

                if sid == -1:
                    continue
                batch_pred_roles[bid].append(((info['trigger span']+(info['event type'],)), (sid, eid, role_type)))

            batch_gold_outputs[bid].append(g_text)
            batch_pred_outputs[bid].append(p_text)

    batch_pred_roles = [sorted(set(role)) for role in batch_pred_roles]
    
    dev_gold_triggers.extend(batch.triggers)
    dev_gold_roles.extend(batch.roles)
    dev_pred_roles.extend(batch_pred_roles)
    dev_pred_wnd_ids.extend(batch.wnd_ids)
    dev_gold_outputs.extend(batch_gold_outputs)
    dev_pred_outputs.extend(batch_pred_outputs)
    dev_inputs.extend(batch_inputs)

progress.close()

# calculate scores
dev_scores = cal_scores(dev_gold_roles, dev_pred_roles)

print("---------------------------------------------------------------------")
print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    dev_scores['arg_id'][3] * 100.0, dev_scores['arg_id'][2], dev_scores['arg_id'][1], 
    dev_scores['arg_id'][4] * 100.0, dev_scores['arg_id'][2], dev_scores['arg_id'][0], dev_scores['arg_id'][5] * 100.0))
print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    dev_scores['arg_cls'][3] * 100.0, dev_scores['arg_cls'][2], dev_scores['arg_cls'][1], 
    dev_scores['arg_cls'][4] * 100.0, dev_scores['arg_cls'][2], dev_scores['arg_cls'][0], dev_scores['arg_cls'][5] * 100.0))
print("---------------------------------------------------------------------")


# write outputs
dev_outputs = {}
for (dev_pred_wnd_id, dev_gold_trigger, dev_gold_role, dev_pred_role, dev_gold_output, dev_pred_output, dev_input) in zip(
    dev_pred_wnd_ids, dev_gold_triggers, dev_gold_roles, dev_pred_roles, dev_gold_outputs, dev_pred_outputs, dev_inputs):
    dev_outputs[dev_pred_wnd_id] = {
        "input": dev_input, 
        "triggers": dev_gold_trigger,
        "gold_roles": dev_gold_role,
        "pred_roles": dev_pred_role,
        "gold_text": dev_gold_output,
        "pred_text": dev_pred_output,
    }

with open(os.path.join(args.output_dir, 'dev.pred.json'), 'w') as fp:
    json.dump(dev_outputs, fp, indent=2)
    
    
# test set
progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test')
test_gold_triggers, test_gold_roles, test_pred_roles = [], [], []
test_pred_wnd_ids, test_gold_outputs, test_pred_outputs, test_inputs = [], [], [], []
for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn):
    progress.update(1)

    batch_pred_roles = [[] for _ in range(config.eval_batch_size)]
    batch_pred_outputs = [[] for _ in range(config.eval_batch_size)]
    batch_gold_outputs = [[] for _ in range(config.eval_batch_size)]
    batch_inputs = [[] for _ in range(config.eval_batch_size)]
    batch_event_templates = []
    for tokens, triggers, roles in zip(batch.tokens, batch.triggers, batch.roles):
        batch_event_templates.append(event_template_generator(tokens, triggers, roles, config.input_style, config.output_style, vocab, config.lang))
    
    # convert EE instances to EAE instances 
    eae_inputs, eae_gold_outputs, eae_events, eae_bids = [], [], [], []
    for i, event_temp in enumerate(batch_event_templates):
        for data in event_temp.get_training_data():
            eae_inputs.append(data[0])
            eae_gold_outputs.append(data[1])
            eae_events.append(data[2])
            eae_bids.append(i)
            batch_inputs[i].append(data[0])
    
    # if there are triggers in this batch, predict argument roles
    if len(eae_inputs) > 0:
        eae_inputs = tokenizer(eae_inputs, return_tensors='pt', padding=True, max_length=config.max_length+2)
        enc_idxs = eae_inputs['input_ids']
        enc_idxs = enc_idxs.cuda()
        enc_attn = eae_inputs['attention_mask'].cuda()

        if config.beam_size == 1:
            model.model._cache_input_ids = enc_idxs
        else:
            expanded_return_idx = (
                torch.arange(enc_idxs.shape[0]).view(-1, 1).repeat(1, config.beam_size).view(-1).to(enc_idxs.device)
            )
            input_ids = enc_idxs.index_select(0, expanded_return_idx)
            model.model._cache_input_ids = input_ids
        
        # inference
        with torch.no_grad():
            if args.constrained_decode:
                prefix_fn_obj = Prefix_fn_cls(tokenizer, ["[and]"], enc_idxs)
                outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                        num_beams=config.beam_size, 
                        max_length=config.max_output_length,
                        forced_bos_token_id=None,
                        prefix_allowed_tokens_fn=lambda batch_id, sent: prefix_fn_obj.get(batch_id, sent)
                        )
            else:
                outputs = model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, 
                    num_beams=config.beam_size, max_length=config.max_output_length, 
                    forced_bos_token_id=None)
        eae_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
        
        # extract argument roles from the generated outputs
        for p_text, g_text, info, bid in zip(eae_pred_outputs, eae_gold_outputs, eae_events, eae_bids):
            theclass = getattr(sys.modules[TEMP_FILE], info['event type'].replace(':', '_').replace('-', '_'), False)
            assert theclass
            template = theclass(config.input_style, config.output_style, info['tokens'], info['event type'], config.lang, info)

            pred_object = template.decode(p_text)

            for span, role_type, _ in pred_object:
                # convert the predicted span to the offsets in the passage
                # Chinese uses a different function since there is no space between Chenise characters
                if config.lang == "chinese":
                    sid, eid = get_span_idxs_zh(batch.tokens[bid], span, trigger_span=info['trigger span'])
                else:
                    sid, eid = get_span_idxs(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer, trigger_span=info['trigger span'])

                if sid == -1:
                    continue
                batch_pred_roles[bid].append(((info['trigger span']+(info['event type'],)), (sid, eid, role_type)))

            batch_gold_outputs[bid].append(g_text)
            batch_pred_outputs[bid].append(p_text)

    batch_pred_roles = [sorted(set(role)) for role in batch_pred_roles]
    
    test_gold_triggers.extend(batch.triggers)
    test_gold_roles.extend(batch.roles)
    test_pred_roles.extend(batch_pred_roles)
    test_pred_wnd_ids.extend(batch.wnd_ids)
    test_gold_outputs.extend(batch_gold_outputs)
    test_pred_outputs.extend(batch_pred_outputs)
    test_inputs.extend(batch_inputs)

progress.close()

# calculate scores
test_scores = cal_scores(test_gold_roles, test_pred_roles)

print("---------------------------------------------------------------------")
print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['arg_id'][3] * 100.0, test_scores['arg_id'][2], test_scores['arg_id'][1], 
    test_scores['arg_id'][4] * 100.0, test_scores['arg_id'][2], test_scores['arg_id'][0], test_scores['arg_id'][5] * 100.0))
print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['arg_cls'][3] * 100.0, test_scores['arg_cls'][2], test_scores['arg_cls'][1], 
    test_scores['arg_cls'][4] * 100.0, test_scores['arg_cls'][2], test_scores['arg_cls'][0], test_scores['arg_cls'][5] * 100.0))
print("---------------------------------------------------------------------")

# write outputs
test_outputs = {}
for (test_pred_wnd_id, test_gold_trigger, test_gold_role, test_pred_role, test_gold_output, test_pred_output, test_input) in zip(
    test_pred_wnd_ids, test_gold_triggers, test_gold_roles, test_pred_roles, test_gold_outputs, test_pred_outputs, test_inputs):
    test_outputs[test_pred_wnd_id] = {
        "input": test_input, 
        "triggers": test_gold_trigger,
        "gold_roles": test_gold_role,
        "pred_roles": test_pred_role,
        "gold_text": test_gold_output,
        "pred_text": test_pred_output,
    }

with open(os.path.join(args.output_dir, 'test.pred.json'), 'w') as fp:
    json.dump(test_outputs, fp, indent=2)
