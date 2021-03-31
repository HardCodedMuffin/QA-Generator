import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random
import spacy
import boto3
import zipfile
import os
import json
from sense2vec import Sense2Vec
import requests
from collections import OrderedDict
import string
import pke
import nltk
from nltk import FreqDist

nltk.download('brown')
nltk.download('stopwords')
nltk.download('popular')
from nltk.corpus import stopwords
from nltk.corpus import brown
from similarity.normalized_levenshtein import NormalizedLevenshtein
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor


def tokenize_sentences(text):
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    # Remove any short sentences less than 20 letters.
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences


def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        word = word.strip()
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values

    delete_keys = []
    for k in keyword_sentences.keys():
        if len(keyword_sentences[k]) == 0:
            delete_keys.append(k)
    for del_key in delete_keys:
        del keyword_sentences[del_key]

    return keyword_sentences


def is_far(words_list, currentword, thresh, normalized_levenshtein):
    threshold = thresh
    score_list = []
    for word in words_list:
        score_list.append(normalized_levenshtein.distance(word.lower(), currentword.lower()))
    if min(score_list) >= threshold:
        return True
    else:
        return False


def filter_phrases(phrase_keys, max, normalized_levenshtein):
    filtered_phrases = []
    if len(phrase_keys) > 0:
        filtered_phrases.append(phrase_keys[0])
        for ph in phrase_keys[1:]:
            if is_far(filtered_phrases, ph, 0.7, normalized_levenshtein):
                filtered_phrases.append(ph)
            if len(filtered_phrases) >= max:
                break
    return filtered_phrases


def get_nouns_multipartite(text):
    out = []

    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text, language='en')
    pos = {'PROPN', 'NOUN'}
    stoplist = list(string.punctuation)
    stoplist += stopwords.words('english')
    extractor.candidate_selection(pos=pos, stoplist=stoplist)
    # 4. build the Multipartite graph and rank candidates using random walk,
    #    alpha controls the weight adjustment mechanism, see TopicRank for
    #    threshold/method parameters.
    try:
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
    except:
        return out

    keyphrases = extractor.get_n_best(n=10)

    for key in keyphrases:
        out.append(key[0])

    return out


def get_phrases(doc):
    phrases = {}
    for np in doc.noun_chunks:
        phrase = np.text
        len_phrase = len(phrase.split())
        if len_phrase > 1:
            if phrase not in phrases:
                phrases[phrase] = 1
            else:
                phrases[phrase] = phrases[phrase] + 1

    phrase_keys = list(phrases.keys())
    phrase_keys = sorted(phrase_keys, key=lambda x: len(x), reverse=True)
    phrase_keys = phrase_keys[:50]
    return phrase_keys


def get_keywords(nlp, text, max_keywords, fdist, normalized_levenshtein, no_of_sentences):
    doc = nlp(text)
    max_keywords = int(max_keywords)

    keywords = get_nouns_multipartite(text)
    keywords = sorted(keywords, key=lambda x: fdist[x])
    keywords = filter_phrases(keywords, max_keywords, normalized_levenshtein)

    phrase_keys = get_phrases(doc)
    filtered_phrases = filter_phrases(phrase_keys, max_keywords, normalized_levenshtein)

    total_phrases = keywords + filtered_phrases

    total_phrases_filtered = filter_phrases(total_phrases, min(max_keywords, 2 * no_of_sentences),
                                            normalized_levenshtein)

    answers = []
    for answer in total_phrases_filtered:
        if answer not in answers:
            answers.append(answer)

    answers = answers[:max_keywords]
    return answers


def generate_normal_questions(keyword_sent_mapping, device, tokenizer, model):  # for normal one word questions
    batch_text = []
    answers = keyword_sent_mapping.keys()
    for answer in answers:
        txt = keyword_sent_mapping[answer]
        context = "context: " + txt
        text = context + " " + "answer: " + answer + " </s>"
        batch_text.append(text)

    encoding = tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True, return_tensors="pt")

    print("Running model for generation")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    with torch.no_grad():
        outs = model.generate(input_ids=input_ids,
                              attention_mask=attention_masks,
                              max_length=150)

    output_array = {}
    output_array["questions"] = []

    for index, val in enumerate(answers):
        individual_quest = {}
        out = outs[index, :]
        dec = tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        Question = dec.replace('question:', '')
        Question = Question.strip()

        individual_quest['Question'] = Question
        individual_quest['Answer'] = val
        individual_quest["id"] = index + 1
        individual_quest["context"] = keyword_sent_mapping[val]

        output_array["questions"].append(individual_quest)

    return output_array


def random_choice():
    a = random.choice([0, 1])
    return bool(a)
