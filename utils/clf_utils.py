"""
Supporting functions for classifiers.
"""

import re, spacy
import utils.bert_utils as bu
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class ContentTypeData:
    """
    train and test data 
    """

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe(self.nlp.create_pipe('merge_entities'))

    def abstract_sents(self, messages):
        """
        abstract sentences using spacy ner tagger
        """
        # print(f'\nInside ContentTypeData.abstract_sents() method\n')
        abs_messages = []
        for idx, message in tqdm(enumerate(messages)):
            message = re.sub('\(\d+-\d+ \w{0,3}, \d+-\d+ \w{0,3}, \d+-\d+ \w{0,3}\)', '(ShotBreakdown)', message)
            message = re.sub('\s+', ' ', message)
            doc = self.nlp(message)
            abs_messages.append(" ".join([t.text if not t.ent_type_ else t.ent_type_ for t in doc]))
        # print(f'\nGoing out of ContentTypeData.abstract_sents() method\n')
        return abs_messages


class TextFeatureExtractor:
    """
    type = [tfidf, tf, bert_emb]
    """

    def __init__(self, type='bert_emb'):
        self.type = type
        self.embedding_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    def extract_train_features(self, messages):
        if self.type == 'bert_emb':
            return self.embedding_model.encode(messages)
        else:
            raise ValueError(f'{self.type} is not supported')

    def extract_test_features(self, messages):
        if self.type == 'bert_emb':
            return self.embedding_model.encode(messages)
        else:
            raise ValueError(f'{self.type} is not supported')


class MultiLabelClassifier:
    """
    model_name = [svm, lr, rf, bert]
    """

    def __init__(self, model_name='bert', ftr_name='bert_embs', model_path='./output/models', dataset_name='sportsett', num_classes=3):
        self.path = model_path
        self.model_name = model_name
        self.ftr_name = ftr_name
        self.save_model_name = f'{self.model_name}_w_{self.ftr_name}'
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.ftr_extractor = TextFeatureExtractor(self.ftr_name) if self.model_name != 'bert' else None

    def train_multilabel_classif(self, messages, labels):
        if self.model_name == 'bert':
            num_epochs = 10
            return bu.train_bert_multilabel_classif(messages, labels, num_epochs=num_epochs, dataset=self.dataset_name, num_classes=self.num_classes, \
                                                    path=f'./{self.dataset_name}/{self.path}/multilabel_roberta.pt')
        else:
            raise ValueError(f'{self.model_name} is not supported')

    def predict_multilabel_classif(self, messages, pred_probs=False):
        if self.model_name == 'bert':
            return bu.predict_bert_multilabel_classif(messages, pred_probs=pred_probs, dataset=self.dataset_name, num_classes=self.num_classes, \
                                                    path=f'./{self.dataset_name}/{self.path}/multilabel_roberta.pt')
        else:
            raise ValueError(f'{self.model_name} is not supported')

