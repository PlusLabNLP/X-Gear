import sys, re

INPUT_STYLE_SET = ['triggerword', 'template']
OUTPUT_STYLE_SET = ['argument:roletype']
NO_SPACE_LANGS = {"chinese"}
IN_SEP = {
    'triggerword': '<--triggerword-->',
    'template': '<--template-->',
}
ROLE_LIST = [
    'Person', 'Entity', 'Defendant', 'Prosecutor', 'Plaintiff', 'Artifact', 'Destination', 
    'Origin', 'Agent', 'Attacker', 'Target', 'Victim', 'Instrument', 'Giver', 'Recipient', 
    'Org', 'Place', 'Adjudicator', 'Beneficiary', 'Thing', 'Audience'
]
NO_ROLE = "[None]"
AND = "[and]"

class event_template_generator():
    def __init__(self, passage, triggers, roles, input_style, output_style, vocab, lang):
        """
        generate strctured information for events
        
        args:
            passage(List): a list of tokens
            triggers(List): a list of triggers
            roles(List): a list of Roles
            input_style(List): List of elements; elements belongs to INPUT_STYLE_SET
            input_style(List): List of elements; elements belongs to OUTPUT_STYLE_SET
        """
        self.raw_passage = passage
        self.lang = lang
        self.no_space = lang in NO_SPACE_LANGS
        self.triggers = triggers
        self.roles = roles
        self.events = self.process_events(passage, triggers, roles)
        self.input_style = input_style
        self.output_style = output_style
        self.vocab = vocab
        self.event_templates = []
        
        for event in self.events:
            theclass = getattr(sys.modules[__name__], event['event type'].replace(':', '_').replace('-', '_'), False)
            assert theclass
            self.event_templates.append(theclass(self.input_style, self.output_style, event['tokens'], event['event type'], self.lang, event))
        
        self.data = [x.generate_pair(x.trigger_text) for x in self.event_templates]
        self.data = [x for x in self.data if x]
        
    def get_training_data(self):
        return self.data

    def process_events(self, passage, triggers, roles):
        """
        Given a list of token and event annotation, return a list of structured event

        structured_event:
        {
            'trigger text': str,
            'trigger span': (start, end),
            'event type': EVENT_TYPE(str),
            'arguments':{
                ROLE_TYPE(str):[{
                    'argument text': str,
                    'argument span': (start, end)
                }],
                ROLE_TYPE(str):...,
                ROLE_TYPE(str):....
            }
            'passage': PASSAGE
        }
        """
        
        events = {trigger: [] for trigger in triggers}

        for argument in roles:
            trigger = argument[0]
            events[trigger].append(argument)
        
        event_structures = []
        for trigger, arguments in events.items():
            eve_type = trigger[2]
            eve_text = ''.join(passage[trigger[0]:trigger[1]]) if self.no_space else ' '.join(passage[trigger[0]:trigger[1]])
            eve_span = (trigger[0], trigger[1])
            argus = {}
            for argument in arguments:
                role_type = argument[1][2]
                if role_type not in argus.keys():
                    argus[role_type] = []
                argus[role_type].append({
                    'argument text': ''.join(passage[argument[1][0]:argument[1][1]]) if self.no_space else ' '.join(passage[argument[1][0]:argument[1][1]]),
                    'argument span': (argument[1][0], argument[1][1]),
                })
            event_structures.append({
                'trigger text': eve_text,
                'trigger span': eve_span,
                'event type': eve_type,
                'arguments': argus,
                'passage': ''.join(passage) if self.no_space else ' '.join(passage),
                'tokens': passage
            })
        return event_structures

class event_template():
    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        self.input_style = input_style
        self.output_style = output_style
        self.output_template = self.get_output_template()
        self.lang = lang
        self.no_space = lang in NO_SPACE_LANGS
        self.passage = ''.join(passage) if self.no_space else ' '.join(passage)
        self.tokens = passage
        self.event_type = event_type
        if gold_event is not None:
            self.gold_event = gold_event
            self.trigger_text = gold_event['trigger text']
            self.trigger_span = [gold_event['trigger span']]
            self.arguments = [gold_event['arguments']]
        else:
            self.gold_event = None
    
    def generate_pair(self, query_trigger):
        """
        Generate model input sentence and output sentence pair
        """
        input_str = self.generate_input_str(query_trigger)
        output_str, gold_sample = self.generate_output_str(query_trigger)
        return (input_str, output_str, self.gold_event, gold_sample, self.event_type, self.tokens)

    def generate_input_str(self, query_trigger):
        input_str = self.passage
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'triggerword':
                    input_str += ' {} {}'.format(IN_SEP['triggerword'], query_trigger)
                if i_style == 'template':
                    input_str += ' {} {}'.format(IN_SEP['template'], self.output_template)
        return input_str

    def generate_output_str(self, query_trigger):
        return (None, False)

    def decode(self, preds):
        output = []
        for cnt, pred in enumerate(preds.split('\n')):
            used_o_cnt = 0
            full_pred = pred.strip()
            for o_style in OUTPUT_STYLE_SET:
                if o_style in self.output_style:
                    if o_style == 'argument:roletype':
                        if used_o_cnt == cnt:
                            try:
                                for a_cnt, prediction in enumerate(full_pred.split(' <sep> ')):
                                    
                                    tag_s = re.search('<--[^/>][^>]*-->', prediction)
                                    while tag_s:
                                        prediction = prediction[tag_s.end():]
                                        r_type = tag_s.group()[3:-3]
                                        
                                        if r_type in ROLE_LIST:
                                            tag_e = re.search(f'</--{r_type}-->', prediction)
                                            if tag_e:
                                                arg = prediction[:tag_e.start()].strip()
                                                for a in arg.split(f' {AND} '):
                                                    a = a.strip()
                                                    if a != '' and a != NO_ROLE:
                                                        output.append((a, r_type, {'cor tri cnt': a_cnt}))
                                                prediction = prediction[tag_e.end():]
                                        
                                        tag_s = re.search('<--[^/>][^>]*-->', prediction)
                            except:
                                pass
                        used_o_cnt += 1

        return output

    def evaluate(self, predict_output):
        assert self.gold_event is not None
        # categorize prediction
        pred_trigger = []
        pred_argument = []
        for pred in predict_output:
            if pred[1] == self.event_type:
                pred_trigger.append(pred)
            else:
                pred_argument.append(pred)
        # trigger score
        gold_tri_num = len(self.trigger_span)
        pred_tris = []
        for pred in pred_trigger:
            pred_span = self.predstr2span(pred[0])
            if pred_span[0] > -1:
                pred_tris.append((pred_span[0], pred_span[1], pred[1]))
        pred_tri_num = len(pred_tris)
        match_tri = 0
        for pred in pred_tris:
            id_flag = False
            for gold_span in self.trigger_span:
                if gold_span[0] == pred[0] and gold_span[1] == pred[1]:
                    id_flag = True
            match_tri += int(id_flag)

        # argument score
        converted_gold = self.get_converted_gold()
        gold_arg_num = len(converted_gold)
        pred_arg = []
        for pred in pred_argument:
            # find corresponding trigger
            pred_span = self.predstr2span(pred[0], self.trigger_span[0][0])
            if (pred_span is not None) and (pred_span[0] > -1):
                pred_arg.append((pred_span[0], pred_span[1], pred[1]))
        pred_arg = list(set(pred_arg))
        pred_arg_num = len(pred_arg)
        
        target = converted_gold
        match_id = 0
        match_type = 0
        for pred in pred_arg:
            id_flag = False
            id_type = False
            for gold in target:
                if gold[0]==pred[0] and gold[1]==pred[1]:
                    id_flag = True
                    if gold[2] == pred[2]:
                        id_type = True
                        break
            match_id += int(id_flag)
            match_type += int(id_type)
        return {
            'gold_tri_num': gold_tri_num, 
            'pred_tri_num': pred_tri_num,
            'match_tri_num': match_tri,
            'gold_arg_num': gold_arg_num,
            'pred_arg_num': pred_arg_num,
            'match_arg_id': match_id,
            'match_arg_cls': match_type
        }
    
    def get_converted_gold(self):
        converted_gold = []
        for argu in self.arguments:
            for arg_type, arg_list in argu.items():
                for arg in arg_list:
                    converted_gold.append((arg['argument span'][0], arg['argument span'][1], arg_type))
        return list(set(converted_gold))
    
    def predstr2span(self, pred_str, trigger_idx=None):
        sub_words = [_.strip() for _ in pred_str.strip().lower().split()]
        candidates=[]
        for i in range(len(self.tokens)):
            j = 0
            while j < len(sub_words) and i+j < len(self.tokens):
                if self.tokens[i+j].lower() == sub_words[j]:
                    j += 1
                else:
                    break
            if j == len(sub_words):
                candidates.append((i, i+len(sub_words)))
        if len(candidates) < 1:
            return -1, -1
        else:
            if trigger_idx is not None:
                return sorted(candidates, key=lambda x: abs(trigger_idx-x[0]))[0]
            else:
                return candidates[0]

class Life_Be_Born(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Person--> [None] </--Person--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Person--> {} </--Person--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Life_Marry(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)
    
    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Person--> [None] </--Person--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Person--> {} </--Person--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Life_Divorce(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)
    
    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Person--> [None] </--Person-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Person--> {} </--Person-->".format(filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Life_Injure(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Agent--> [None] </--Agent--> <--Victim--> [None] </--Victim--> <--Instrument--> [None] </--Instrument--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Victim']]) if "Victim" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Instrument']]) if "Instrument" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Agent--> {} </--Agent--> <--Victim--> {} </--Victim--> <--Instrument--> {} </--Instrument--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Life_Die(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)
    
    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Agent--> [None] </--Agent--> <--Victim--> [None] </--Victim--> <--Instrument--> [None] </--Instrument--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Victim']]) if "Victim" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Instrument']]) if "Instrument" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Agent--> {} </--Agent--> <--Victim--> {} </--Victim--> <--Instrument--> {} </--Instrument--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Transaction_Transfer_Ownership(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Giver--> [None] </--Giver--> <--Recipient--> [None] </--Recipient--> <--Place--> [None] </--Place--> <--Beneficiary--> [None] </--Beneficiary--> <--Thing--> [None] </--Thing-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Giver']]) if "Giver" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Recipient']]) if "Recipient" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Beneficiary']]) if "Beneficiary" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Thing']]) if "Thing" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Giver--> {} </--Giver--> <--Recipient--> {} </--Recipient--> <--Place--> {} </--Place--> <--Beneficiary--> {} </--Beneficiary--> <--Thing--> {} </--Thing-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Transaction_Transfer_Money(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Giver--> [None] </--Giver--> <--Recipient--> [None] </--Recipient--> <--Place--> [None] </--Place--> <--Beneficiary--> [None] </--Beneficiary-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Giver']]) if "Giver" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Recipient']]) if "Recipient" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Beneficiary']]) if "Beneficiary" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Giver--> {} </--Giver--> <--Recipient--> {} </--Recipient--> <--Place--> {} </--Place--> <--Beneficiary--> {} </--Beneficiary-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Transaction_Transaction(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Giver--> [None] </--Giver--> <--Recipient--> [None] </--Recipient--> <--Place--> [None] </--Place--> <--Beneficiary--> [None] </--Beneficiary-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Giver']]) if "Giver" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Recipient']]) if "Recipient" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Beneficiary']]) if "Beneficiary" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Giver--> {} </--Giver--> <--Recipient--> {} </--Recipient--> <--Place--> {} </--Place--> <--Beneficiary--> {} </--Beneficiary-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Business_Start_Org(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Agent--> [None] </--Agent--> <--Org--> [None] </--Org--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Org']]) if "Org" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Agent--> {} </--Agent--> <--Org--> {} </--Org--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Business_Merge_Org(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Org--> [None] </--Org-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Org']]) if "Org" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Org--> {} </--Org-->".format(filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Business_Declare_Bankruptcy(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Org--> [None] </--Org--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Org']]) if "Org" in argu.keys() else NO_ROLE, 
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Org--> {} </--Org--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Business_End_Org(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Org--> [None] </--Org--> <--Place--> [None] </--Place-->') # Agent is mislabel.
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Org']]) if "Org" in argu.keys() else NO_ROLE, 
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Org--> {} </--Org--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    
class Conflict_Attack(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Attacker--> [None] </--Attacker--> <--Target--> [None] </--Target--> <--Instrument--> [None] </--Instrument--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        attacker_list = []
                        if "Attacker" in argu.keys():
                            attacker_list.extend([ a['argument text'] for a in argu['Attacker']])
                        if "Agent" in argu.keys():
                            attacker_list.extend([ a['argument text'] for a in argu['Agent']])  
                        filler = (
                            f" {AND} ".join(attacker_list) if len(attacker_list)>0 else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Target']]) if "Target" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Instrument']]) if "Instrument" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Attacker--> {} </--Attacker--> <--Target--> {} </--Target--> <--Instrument--> {} </--Instrument--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Conflict_Demonstrate(event_template):
    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Entity--> [None] </--Entity--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Entity--> {} </--Entity--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Contact_Contact(event_template):
    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Entity--> [None] </--Entity--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Entity--> {} </--Entity--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    
class Contact_Correspondence(event_template):
    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Entity--> [None] </--Entity--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Entity--> {} </--Entity--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Contact_Broadcast(event_template):
    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Entity--> [None] </--Entity--> <--Place--> [None] </--Place--> <--Audience--> [None] </--Audience-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE, 
                            f" {AND} ".join([ a['argument text'] for a in argu['Audience']]) if "Audience" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Entity--> {} </--Entity--> <--Place--> {} </--Place--> <--Audience--> {} </--Audience-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Contact_Meet(event_template):
    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Entity--> [None] </--Entity--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Entity--> {} </--Entity--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

            
class Personnel_Start_Position(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Person--> [None] </--Person--> <--Entity--> [None] </--Entity--> <--Place--> [None] </--Place-->') #Agent is mislabel
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Person--> {} </--Person--> <--Entity--> {} </--Entity--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    
class Personnel_End_Position(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Person--> [None] </--Person--> <--Entity--> [None] </--Entity--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Person--> {} </--Person--> <--Entity--> {} </--Entity--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Personnel_Nominate(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Person--> [None] </--Person--> <--Agent--> [None] </--Agent--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else NO_ROLE, 
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Person--> {} </--Person--> <--Agent--> {} </--Agent--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Personnel_Elect(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)
    
    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Person--> [None] </--Person--> <--Agent--> [None] </--Agent--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Person--> {} </--Person--> <--Agent--> {} </--Agent--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Justice_Arrest_Jail(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Person--> [None] </--Person--> <--Agent--> [None] </--Agent--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Person--> {} </--Person--> <--Agent--> {} </--Agent--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    
class Justice_Release_Parole(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Person--> [None] </--Person--> <--Agent--> [None] </--Agent--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Person--> {} </--Person--> <--Agent--> {} </--Agent--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Justice_Trial_Hearing(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)
    
    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Defendant--> [None] </--Defendant--> <--Prosecutor--> [None] </--Prosecutor--> <--Place--> [None] </--Place--> <--Adjudicator--> [None] </--Adjudicator-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Prosecutor']]) if "Prosecutor" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Defendant--> {} </--Defendant--> <--Prosecutor--> {} </--Prosecutor--> <--Place--> {} </--Place--> <--Adjudicator--> {} </--Adjudicator-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Justice_Charge_Indict(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Defendant--> [None] </--Defendant--> <--Prosecutor--> [None] </--Prosecutor--> <--Place--> [None] </--Place--> <--Adjudicator--> [None] </--Adjudicator-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Prosecutor']]) if "Prosecutor" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Defendant--> {} </--Defendant--> <--Prosecutor--> {} </--Prosecutor--> <--Place--> {} </--Place--> <--Adjudicator--> {} </--Adjudicator-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    
class Justice_Sue(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Defendant--> [None] </--Defendant--> <--Plaintiff--> [None] </--Plaintiff--> <--Place--> [None] </--Place--> <--Adjudicator--> [None] </--Adjudicator-->') # Prosecutor is mislabel. 
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Plaintiff']]) if "Plaintiff" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Defendant--> {} </--Defendant--> <--Plaintiff--> {} </--Plaintiff--> <--Place--> {} </--Place--> <--Adjudicator--> {} </--Adjudicator-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    
class Justice_Convict(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Defendant--> [None] </--Defendant--> <--Place--> [None] </--Place--> <--Adjudicator--> [None] </--Adjudicator-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Defendant--> {} </--Defendant--> <--Place--> {} </--Place--> <--Adjudicator--> {} </--Adjudicator-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    
class Justice_Sentence(event_template):
    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Defendant--> [None] </--Defendant--> <--Place--> [None] </--Place--> <--Adjudicator--> [None] </--Adjudicator-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Defendant--> {} </--Defendant--> <--Place--> {} </--Place--> <--Adjudicator--> {} </--Adjudicator-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Justice_Fine(event_template):
    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Entity--> [None] </--Entity--> <--Place--> [None] </--Place--> <--Adjudicator--> [None] </--Adjudicator-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Entity']]) if "Entity" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Entity--> {} </--Entity--> <--Place--> {} </--Place--> <--Adjudicator--> {} </--Adjudicator-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Justice_Execute(event_template):
    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Person--> [None] </--Person--> <--Agent--> [None] </--Agent--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Person--> {} </--Person--> <--Agent--> {} </--Agent--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

            
class Justice_Extradite(event_template):
    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Person--> [None] </--Person--> <--Destination--> [None] </--Destination--> <--Origin--> [None] </--Origin--> <--Agent--> [None] </--Agent-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Destination']]) if "Destination" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Origin']]) if "Origin" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Person--> {} </--Person--> <--Destination--> {} </--Destination--> <--Origin--> {} </--Origin--> <--Agent--> {} </--Agent-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Justice_Acquit(event_template):
    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Defendant--> [None] </--Defendant--> <--Place--> [None] </--Place--> <--Adjudicator--> [None] </--Adjudicator-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Defendant--> {} </--Defendant--> <--Place--> {} </--Place--> <--Adjudicator--> {} </--Adjudicator-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    
class Justice_Pardon(event_template):
    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Defendant--> [None] </--Defendant--> <--Place--> [None] </--Place--> <--Adjudicator--> [None] </--Adjudicator-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Defendant--> {} </--Defendant--> <--Place--> {} </--Place--> <--Adjudicator--> {} </--Adjudicator-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    
class Justice_Appeal(event_template):
    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Defendant--> [None] </--Defendant--> <--Prosecutor--> [None] </--Prosecutor--> <--Place--> [None] </--Place--> <--Adjudicator--> [None] </--Adjudicator-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Defendant']]) if "Defendant" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Prosecutor']]) if "Prosecutor" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Adjudicator']]) if "Adjudicator" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Defendant--> [None] </--Defendant--> <--Prosecutor--> [None] </--Prosecutor--> <--Place--> {} </--Place--> <--Adjudicator--> {} </--Adjudicator-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    
class Manufacture_Artifact(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)
    
    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Artifact--> [None] </--Artifact--> <--Agent--> [None] </--Agent--> <--Place--> [None] </--Place-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Artifact']]) if "Artifact" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Place']]) if "Place" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Artifact--> {} </--Artifact--> <--Agent--> {} </--Agent--> <--Place--> {} </--Place-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)

    
class Movement_Transport_Artifact(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Artifact--> [None] </--Artifact--> <--Destination--> [None] </--Destination--> <--Origin--> [None] </--Origin--> <--Agent--> [None] </--Agent--> <--Instrument--> [None] </--Instrument-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Artifact']]) if "Artifact" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Destination']]) if "Destination" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Origin']]) if "Origin" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Instrument']]) if "Instrument" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Artifact--> {} </--Artifact--> <--Destination--> {} </--Destination--> <--Origin--> {} </--Origin--> <--Agent--> {} </--Agent--> <--Instrument--> {} </--Instrument-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)


class Movement_Transport_Person(event_template):

    def __init__(self, input_style, output_style, passage, event_type, lang, gold_event=None):
        super().__init__(input_style, output_style, passage, event_type, lang, gold_event)

    def get_output_template(self):
        output_template = ''
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_template += ' \n {}'.format('<--Person--> [None] </--Person--> <--Destination--> [None] </--Destination--> <--Origin--> [None] </--Origin--> <--Agent--> [None] </--Agent--> <--Instrument--> [None] </--Instrument-->')
        return ('\n'.join(output_template.split('\n')[1:])).strip()

    def generate_output_str(self, query_trigger):
        assert self.gold_event is not None
        output_str = ''
        gold_sample = False
        for o_style in OUTPUT_STYLE_SET:
            if o_style in self.output_style:
                if o_style == 'argument:roletype':
                    output_texts = []
                    for argu in self.arguments:
                        filler = (
                            f" {AND} ".join([ a['argument text'] for a in argu['Person']]) if "Person" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Destination']]) if "Destination" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Origin']]) if "Origin" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Agent']]) if "Agent" in argu.keys() else NO_ROLE,
                            f" {AND} ".join([ a['argument text'] for a in argu['Instrument']]) if "Instrument" in argu.keys() else NO_ROLE
                        )
                        output_texts.append("<--Person--> {} </--Person--> <--Destination--> {} </--Destination--> <--Origin--> {} </--Origin--> <--Agent--> {} </--Agent--> <--Instrument--> {} </--Instrument-->".format(*filler))
                        gold_sample = True
                    output_str += ' \n {}'.format(' <sep> '.join(output_texts))
        output_str = ('\n'.join(output_str.split('\n')[1:])).strip()
        return (output_str, gold_sample)
