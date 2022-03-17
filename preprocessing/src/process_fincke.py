import argparse
import json
from collections import Counter, defaultdict
from transformers import (BertTokenizer,
                          RobertaTokenizer,
                          XLMRobertaTokenizer,
                          PreTrainedTokenizer,
                          BartTokenizer,
                          MBart50Tokenizer,
                          MT5Tokenizer,
                          AutoTokenizer)
import ipdb
import stanza
import torch
import numpy as np

nlp_ar_dep = stanza.Pipeline(lang='ar', tokenize_pretokenized=True)
nlp_en_dep = stanza.Pipeline(lang='en', tokenize_pretokenized=True)
nlp_zh_dep = stanza.Pipeline(lang='zh', tokenize_pretokenized=True)

def distance_matrix(tokens, stanza_result):
    n_tokens = len(tokens)
    parents = np.zeros(n_tokens, dtype=np.long)
    for item in stanza_result[0]:
        i1 = item['id'] - 1
        i2 = item['head'] - 1
        parents[i1] = i2
        
    hights = np.zeros(n_tokens, dtype=np.long)
    for i in range(n_tokens):
        p = i
        while parents[p] != -1:
            hights[i] += 1
            p = parents[p]
            
    dist = np.zeros((n_tokens, n_tokens), dtype=np.long)
    for i in range(n_tokens):
        for j in range(n_tokens):
            d = 0
            pi = i
            pj = j
            while pi != pj:
                d += 1
                if hights[pi] > hights[pj]:
                    pi = parents[pi]
                else:
                    pj = parents[pj]
            dist[i, j] = d
            
    for i in range(n_tokens):
        dist[i, i] = 1
        
    assert not np.any(dist == 0)

    return dist.tolist()

def _distance_matrix(tokens, stanza_result):
    n_tokens = len(tokens)
    links = torch.zeros((n_tokens, n_tokens), dtype=torch.long)
    
    for item in stanza_result[0]:
        if item['head'] > 0:
            i1 = item['id'] - 1
            i2 = item['head'] - 1
            links[i1][i2] = 1
            links[i2][i1] = 1
    
    fake_inf = 99999
    dist = links.clone()
    dist.masked_fill_(1-dist, fake_inf)
    for i in range(n_tokens):
        dist[i][i] = 0
    
    for k in range(n_tokens):
        for i in range(n_tokens):
            for j in range(n_tokens):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    
    assert not torch.any(dist == fake_inf)
    for i in range(n_tokens):
        dist[i][i] = 1
    return dist.cpu().tolist()

def process_doc(data, tokenizer, language, add_stanza):
    output = [] # each document can be split into several windows of data.
    for sentence in data['sentences']:
        res = {}
        res['doc_id'] = data['doc_id']
        if sentence['original_sent_id'] is not None:
            res['wnd_id'] = data['doc_id'] + '_{}_{}'.format(sentence['original_sent_id'], sentence['sent_split_index'])
        else:
            res['wnd_id'] = data['doc_id'] + '_{}'.format(sentence['sent_id'])
        res['sentence_starts'] = [0]
        res['tokens'] = [t['text'] for t in sentence['tokens']]
        res['tokens_info'] = [t for t in sentence['tokens']]
        pieces = [tokenizer.tokenize(t) for t in res['tokens']]
        token_lens = [len(x) for x in pieces]
        if 0 in token_lens:
            print('Throw away one instance!')
            continue #throw this instance away, this only happen in training set

        pieces = [p for ps in pieces for p in ps]
        res['pieces'] = pieces
        res['token_lens'] = token_lens
        res['sentences'] = sentence['original_document_character_span']['text']

        sentence_ner = []
        entity_coref = []
        for mention in sentence['mentions']:
            sentence_ner.append({
                'id': '{}-E{}'.format(data['doc_id'], mention['mention_id']),
                'text': mention['grounded_span']['full_span']['text'],
                'entity_type': mention['entity_type'],
                'mention_type': "UNK" if mention['mention_type'] == '' else mention['mention_type'],
                'start': mention['grounded_span']['full_span']['start_token'],
                'end': mention['grounded_span']['full_span']['end_token']+1,
                'start_char': mention['grounded_span']['full_span']['start_char'],
                'end_char': mention['grounded_span']['full_span']['end_char'] # note that for char span, we didn't add 1.
            })
        
        sentence_events = []
        event_coref = []
        for event in sentence['basic_events']:
            assert len(event['anchors']['spans']) == 1
            eve = {
                'id': '{}-EV{}'.format(data['doc_id'], event['event_id']),
                'event_type': event['event_type'].replace('.', ':', 1),
                'trigger': {
                    'text': event['anchors']['spans'][0]['text'],
                    'start': event['anchors']['spans'][0]['grounded_span']['full_span']['start_token'],
                    'end': event['anchors']['spans'][0]['grounded_span']['full_span']['end_token']+1,
                    'start_char': event['anchors']['spans'][0]['grounded_span']['full_span']['start_char'],
                    'end_char': event['anchors']['spans'][0]['grounded_span']['full_span']['end_char']+1,
                },
                'arguments': []
            }
            for argument in event['arguments']:
                eve['arguments'].append({
                    'entity_id': '{}-E{}'.format(data['doc_id'], argument['span_set']['spans'][0]['grounded_span']['mention_id']),
                    'text': argument['span_set']['spans'][0]['text'],
                    'role': argument['role']
                })
            sentence_events.append(eve)
        res['entity_mentions'] = sentence_ner
        res['relation_mentions'] = []
        res['event_mentions'] = sentence_events
        res['event_coreference'] = []
        res['entity_coreference'] = []
        res['language'] = language
        
        if add_stanza:
            # dependency parsing
            if language == 'english':
                nlp = nlp_en_dep
            elif language == 'arabic':
                nlp = nlp_ar_dep
            elif language == 'chinese':
                nlp = nlp_zh_dep

            try:
                result = nlp([res['tokens']]).to_dict()
            except:
                print('Error in Stanza!!!!!!!!!!')
                continue
            assert len(result) == 1
            assert len(result[0]) == len(res['tokens'])
            for r, t in zip(result[0], res['tokens']):
                assert (r['text'] == t)
                del r['lemma']
                del r['misc']
            res['stanza_result'] = result
            dist = distance_matrix(res['tokens'], result)
            res['dist_matrix'] = dist
        output.append(res)
    
    return output

def process(input_path, output_path, tokenzier, language, add_stanza=False):
    with open(input_path, 'r') as f:
        data = json.load(f)

    processed_data = [process_doc(doc, tokenzier, language, add_stanza) for doc_id, doc in data.items()]

    with open(output_path, 'w') as f:
        for doc in processed_data:
            for sentence in doc:
                f.write(json.dumps(sentence) +'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument('-b', '--bert',
                        help='BERT model name',
                        default='bert-large-cased')
    parser.add_argument('-c', '--bert_cache_dir',
                        help='Path to the BERT cahce directory')
    parser.add_argument('-l', '--lang', default='english', choices=['english', 'arabic'],
                        help='Document language')
    parser.add_argument('--add_stanza', default=False, action='store_true')
    args = parser.parse_args()
        
    # Create a tokenizer based on the model name
    model_name = args.bert
    cache_dir = args.bert_cache_dir
    if model_name.startswith('bert-base-multilingual-cased'):
        tokenizer = BertTokenizer.from_pretrained(model_name)
    elif model_name.startswith('bert-'):
        tokenizer = BertTokenizer.from_pretrained(model_name,
                                                  cache_dir=cache_dir)
    elif model_name.startswith('roberta-'):
        tokenizer = RobertaTokenizer.from_pretrained(model_name,
                                                     cache_dir=cache_dir)
    elif model_name.startswith('xlm-roberta-'):
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name,
                                                        cache_dir=cache_dir)
    elif model_name.startswith('lanwuwei'):
        tokenizer = BertTokenizer.from_pretrained(model_name,
                                                cache_dir=cache_dir, 
                                                do_lower_case=True)
    elif model_name.startswith('google/mt5-'):
        tokenizer = MT5Tokenizer.from_pretrained(model_name,
                                                cache_dir=cache_dir)                                                
    elif model_name.startswith('facebook/bart'):
        tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    elif model_name.startswith('facebook/mbart-'):
        if args.lang == 'english':
            src_lang = "en_XX"
        elif args.lang == 'arabic':
            src_lang = "ar_AR"
        tokenizer = MBart50Tokenizer.from_pretrained(model_name, cache_dir=cache_dir, src_lang=src_lang, tgt_lang="en_XX")   
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    process(args.input_path, args.output_path, tokenizer, args.lang, add_stanza=args.add_stanza)
