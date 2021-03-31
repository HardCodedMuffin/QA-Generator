import time
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import spacy
import nltk
import numpy
from nltk import FreqDist

from nltk.corpus import brown
from similarity.normalized_levenshtein import NormalizedLevenshtein
from Questgen.mcq.mcq import tokenize_sentences
from Questgen.mcq.mcq import get_keywords
from Questgen.mcq.mcq import get_sentences_for_keyword
from Questgen.mcq.mcq import generate_normal_questions

nltk.download('brown')
nltk.download('stopwords')
nltk.download('popular')


class QGen:

    def __init__(self):

        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('Parth/result')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # model.eval()
        self.device = device
        self.model = model
        self.nlp = spacy.load('en_core_web_sm')
        self.fdist = FreqDist(brown.words())
        self.normalized_levenshtein = NormalizedLevenshtein()
        self.set_seed(42)

    def set_seed(self, seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def predict_shortq(self, payload):
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions", 4)
        }

        text = inp['input_text']
        sentences = tokenize_sentences(text)
        joiner = " "
        modified_text = joiner.join(sentences)

        keywords = get_keywords(self.nlp, modified_text, inp['max_questions'], self.fdist,
                                self.normalized_levenshtein, len(sentences))

        keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)

        for k in keyword_sentence_mapping.keys():
            text_snippet = " ".join(keyword_sentence_mapping[k][:3])
            keyword_sentence_mapping[k] = text_snippet

        final_output = {}

        if len(keyword_sentence_mapping.keys()) == 0:
            print('ZERO')
            return final_output
        else:

            generated_questions = generate_normal_questions(keyword_sentence_mapping, self.device, self.tokenizer,
                                                            self.model)
            print(generated_questions)

        final_output["statement"] = modified_text
        final_output["questions"] = generated_questions["questions"]

        if torch.device == 'cuda':
            torch.cuda.empty_cache()

        return final_output


class AnswerPredictor:

    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('Parth/boolean')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # model.eval()
        self.device = device
        self.model = model
        self.set_seed(42)

    def set_seed(self, seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def greedy_decoding(inp_ids, attn_mask, model, tokenizer):
        greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
        Question = tokenizer.decode(greedy_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return Question.strip().capitalize()

    def predict_answer(self, payload):
        start = time.time()
        inp = {
            "input_text": payload.get("input_text"),
            "input_question": payload.get("input_question")
        }

        context = inp["input_text"]
        question = inp["input_question"]
        input = "question: %s <s> context: %s </s>" % (question, context)

        encoding = self.tokenizer.encode_plus(input, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
        greedy_output = self.model.generate(input_ids=input_ids, attention_mask=attention_masks, max_length=256)
        Question = self.tokenizer.decode(greedy_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output = Question.strip().capitalize()

        return output
