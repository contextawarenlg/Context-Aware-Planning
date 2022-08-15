import spacy, neuralcoref, re

class CorefResolver:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        neuralcoref.add_to_pipe(self.nlp)

    def raw_sentence(self, sents):
        return ' '.join([sent.text for sent in sents])

    def spacy_sent_tokenize(self, doc):
        sents = []
        all_sents = []
        valid_stop = False
        for sent in doc.sents:
            sents.append(sent)
            valid_stop = True if sent[-1].text in ['.', '?', '!'] else False
            if valid_stop:
                all_sents.append(self.raw_sentence(sents))
                sents = []
        return all_sents

    def process_one_summary(self, summary):
        if summary != '':
            summary = re.sub('\(\d+-\d+ \w{0,3}, \d+-\d+ \w{0,3}, \d+-\d+ \w{0,3}\)', '(ShotBreakdown)', summary)
            summary = re.sub('\s+', ' ', summary)

            orig_doc = self.nlp(summary)
            doc = self.nlp(orig_doc._.coref_resolved)

            sents         = []
            sent_num      = 0
            valid_stop    = False
            abs_sents     = []

            for sent in doc.sents:
                sents.append(sent)
                valid_stop = True if sent[-1].text in ['.', '?', '!'] else False
                if valid_stop:
                    raw_sentence = self.raw_sentence(sents)
                    abs_sents.append(raw_sentence)
                    sents = []
                    sent_num += 1
        return abs_sents
