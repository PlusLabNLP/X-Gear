import os, sys, json, logging, time, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import MT5Tokenizer, AdamW, get_linear_schedule_with_warmup
from model import GenerativeModel
from data import GenDataset
from utils import Summarizer, compute_f1
from argparse import ArgumentParser, Namespace

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True)
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config.update(args.__dict__)
config = Namespace(**config)

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

# logger and summarizer
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.output_dir, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_path = os.path.join(output_dir, "train.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', 
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
summarizer = Summarizer(output_dir)

# check valid styles
assert np.all([style in ['triggerword', 'template'] for style in config.input_style])
assert np.all([style in ['argument:roletype'] for style in config.output_style])

# output
with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
    json.dump(vars(config), fp, indent=4)
best_model_path = os.path.join(output_dir, 'best_model.mdl')
dev_prediction_path = os.path.join(output_dir, 'pred.dev.json')
test_prediction_path = os.path.join(output_dir, 'pred.test.json')

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
train_set = GenDataset(tokenizer, sep_tokens, config.max_length, config.train_finetune_file, config.max_output_length)
dev_set = GenDataset(tokenizer, sep_tokens, config.max_length, config.dev_finetune_file, config.max_output_length)
test_set = GenDataset(tokenizer, sep_tokens, config.max_length, config.test_finetune_file, config.max_output_length)
train_batch_num = len(train_set) // config.train_batch_size + (len(train_set) % config.train_batch_size != 0)
dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)

# initialize the model
model = GenerativeModel(config, tokenizer)
model.cuda(device=config.gpu_device)

# optimizer
param_groups = [{'params': model.parameters(), 'lr': config.learning_rate, 'weight_decay': config.weight_decay}]
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=train_batch_num*config.warmup_epoch,
                                           num_training_steps=train_batch_num*config.max_epoch)


# start training
logger.info("Start training ...")
summarizer_step = 0
best_dev_epoch = -1
best_dev_scores = {
    'arg_id': (0.0, 0.0, 0.0),
    'arg_cls': (0.0, 0.0, 0.0)
}
for epoch in range(1, config.max_epoch+1):
    logger.info(log_path)
    logger.info(f"Epoch {epoch}")
    
    # training
    progress = tqdm.tqdm(total=train_batch_num, ncols=75, desc='Train {}'.format(epoch))
    model.train()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(DataLoader(train_set, batch_size=config.train_batch_size // config.accumulate_step, 
                                                 shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)):
        
        # forard model
        loss = model(batch)
        
        # record loss
        summarizer.scalar_summary('train/loss', loss, summarizer_step)
        summarizer_step += 1
        
        loss = loss * (1 / config.accumulate_step)
        loss.backward()

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            
    progress.close()

    # eval dev set
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev {}'.format(epoch))
    model.eval()
    best_dev_flag = False
    write_output = []
    dev_gold_arg_num, dev_pred_arg_num, dev_match_arg_id, dev_match_arg_cls = 0, 0, 0, 0
    
    for batch_idx, batch in enumerate(DataLoader(dev_set, batch_size=config.eval_batch_size, 
                                                 shuffle=False, collate_fn=dev_set.collate_fn)):
        progress.update(1)
        pred_text = model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
        gold_text = batch.target_text
        input_text = batch.input_text
        for i_text, g_text, p_text, info in zip(input_text, gold_text, pred_text, batch.infos):
            theclass = getattr(sys.modules[TEMP_FILE], info['event type'].replace(':', '_').replace('-', '_'), False)
            assert theclass
            template = theclass(config.input_style, config.output_style, info['tokens'], info['event type'], config.lang, info)
            
            # decode predictions
            pred_object = template.decode(p_text)
            gold_object = template.get_converted_gold()
            
            # calculate scores
            sub_scores = template.evaluate(pred_object)
            dev_gold_arg_num += sub_scores['gold_arg_num']
            dev_pred_arg_num += sub_scores['pred_arg_num']
            dev_match_arg_id += sub_scores['match_arg_id']
            dev_match_arg_cls += sub_scores['match_arg_cls']
            write_output.append({
                'input text': i_text, 
                'gold text': g_text,
                'pred text': p_text,
                'gold arguments': gold_object,
                'pred arguments': pred_object,
                'scores': sub_scores,
                'gold info': info
            })
    progress.close()
    
    dev_scores = {
        'arg_id': compute_f1(dev_pred_arg_num, dev_gold_arg_num, dev_match_arg_id),
        'arg_cls': compute_f1(dev_pred_arg_num, dev_gold_arg_num, dev_match_arg_cls)
    }

    # print scores
    print("---------------------------------------------------------------------")
    print('Role I     - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
        dev_scores['arg_id'][0] * 100.0, dev_match_arg_id, dev_pred_arg_num, 
        dev_scores['arg_id'][1] * 100.0, dev_match_arg_id, dev_gold_arg_num, dev_scores['arg_id'][2] * 100.0))
    print('Role C     - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
        dev_scores['arg_cls'][0] * 100.0, dev_match_arg_cls, dev_pred_arg_num, 
        dev_scores['arg_cls'][1] * 100.0, dev_match_arg_cls, dev_gold_arg_num, dev_scores['arg_cls'][2] * 100.0))
    print("---------------------------------------------------------------------")
    
    # check best dev model
    if dev_scores['arg_cls'][2] >= best_dev_scores['arg_cls'][2]:
        best_dev_flag = True
        
    # if best dev, save model and evaluate test set
    if best_dev_flag:    
        best_dev_scores = dev_scores
        best_dev_epoch = epoch
        
        # save best model
        logger.info('Saving best model')
        torch.save(model.state_dict(), best_model_path)
        
        # save dev result
        with open(dev_prediction_path, 'w') as fp:
            json.dump(write_output, fp, indent=4)

        # eval test set
        progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test {}'.format(epoch))
        write_output = []
        test_gold_arg_num, test_pred_arg_num, test_match_arg_id, test_match_arg_cls = 0, 0, 0, 0
        for batch_idx, batch in enumerate(DataLoader(test_set, batch_size=config.eval_batch_size, 
                                                     shuffle=False, collate_fn=test_set.collate_fn)):
            progress.update(1)
            pred_text = model.predict(batch, num_beams=config.beam_size, max_length=config.max_output_length)
            gold_text = batch.target_text
            input_text = batch.input_text
            for i_text, g_text, p_text, info in zip(input_text, gold_text, pred_text, batch.infos):
                theclass = getattr(sys.modules[TEMP_FILE], info['event type'].replace(':', '_').replace('-', '_'), False)
                assert theclass
                template = theclass(config.input_style, config.output_style, info['tokens'], info['event type'], config.lang, info)
                
                # decode predictions
                pred_object = template.decode(p_text)
                gold_object = template.get_converted_gold()
                
                # calculate scores
                sub_scores = template.evaluate(pred_object)
                test_gold_arg_num += sub_scores['gold_arg_num']
                test_pred_arg_num += sub_scores['pred_arg_num']
                test_match_arg_id += sub_scores['match_arg_id']
                test_match_arg_cls += sub_scores['match_arg_cls']
                write_output.append({
                    'input text': i_text,
                    'gold text': g_text,
                    'pred text': p_text,
                    'gold arguments': gold_object,
                    'pred arguments': pred_object,
                    'scores': sub_scores,
                    'gold info': info
                })
        progress.close()
        
        test_scores = {
            'arg_id': compute_f1(test_pred_arg_num, test_gold_arg_num, test_match_arg_id),
            'arg_cls': compute_f1(test_pred_arg_num, test_gold_arg_num, test_match_arg_cls)
        }
        
        # print scores
        print("---------------------------------------------------------------------")
        print('Role I     - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
            test_scores['arg_id'][0] * 100.0, test_match_arg_id, test_pred_arg_num, 
            test_scores['arg_id'][1] * 100.0, test_match_arg_id, test_gold_arg_num, test_scores['arg_id'][2] * 100.0))
        print('Role C     - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
            test_scores['arg_cls'][0] * 100.0, test_match_arg_cls, test_pred_arg_num, 
            test_scores['arg_cls'][1] * 100.0, test_match_arg_cls, test_gold_arg_num, test_scores['arg_cls'][2] * 100.0))
        print("---------------------------------------------------------------------")
        
        # save test result
        with open(test_prediction_path, 'w') as fp:
            json.dump(write_output, fp, indent=4)
            
    logger.info({"epoch": epoch, "dev_scores": dev_scores})
    if best_dev_flag:
        logger.info({"epoch": epoch, "test_scores": test_scores})
    logger.info("Current best")
    logger.info({"best_epoch": best_dev_epoch, "best_scores": best_dev_scores})
        
logger.info(log_path)
logger.info("Done!")
