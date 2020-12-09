import math
import time
import pandas as pd
import numpy as np
import string

import _thread
import threading

import requests
import xml.dom.minidom
from scipy import spatial
import gensim

output_file = "cloze_predict_35_FULL.tsv"
process_date = "2020_11_26"


def clean_word(word):
    w = "%s" % word
    w = w.lower()
    w = w.replace('”', '')
    w = w.replace('“', '')
    w = w.replace('"', '')
    w = w.replace('...', '')
    w = w.replace('%', '')
    w = w.replace('?', '')
    w = w.replace('!', '')
    w = w.replace(']', '')
    w = w.replace('[', '')
    w = w.replace(')', '')
    w = w.replace('(', '')
    w = w.replace('\\', '')
    # w = w.replace(' ', '')
    w = w.replace(' ', '_')  # \xa0
    w = w.strip()
    w = w.strip(',')
    w = w.strip('.')
    return w


def load_main_dataset():
    df_all = pd.read_csv('data/cloze_all_%s.csv' % process_date, sep='\t', quoting=3)
    print("total:", df_all['Palavra'].count())

    return df_all


def write_output_header(f):
    header = 'Participant\tWord_Unique_ID\tText_ID\tWord_Number\tSentence_Number\tWord_In_Sentence_Number\t' \
             'Word_Place_In_Sent\tSent_Length\tWord\tWord_Cleaned\tWord_Length\tResponse\tResponse_Cleaned\t' \
             'OrthographicMatch\t' \
             'POS\tWord_Content_Or_Function\tWord_POS\tResponse_Tag\tPOSMatch\t' \
             'Word_Inflection\tResponse_Inflection\tInflectionMatch\t' \
             'Freq_brWaC_fpm\tFreq_brWaC_log\tFreq_brasileiro_fpm\tFreq_brasileiro_log\tGenre\t' \
             'Semantic_Word_Context_Score\tSemantic_Response_Match_Score\tSemantic_Response_Context_Score\t' \
             'Semantic_Word_Context_Score_ft\tSemantic_Response_Match_Score_ft\tSemantic_Response_Context_Score_ft\t' \
             'Time_to_Start\tTyping_Time\tTotal_time\t' \
             'Word_Lemma_\tAnswer_Lemma\n'
    print(header)
    f.write(header)


def write_output_line(f, part, word_id, text_id, widx, sent, wsidx, word_place_in_sent, sent_length, w_raw, w, a_raw, a,
                      ortho_match, tag, content_func, tag_desc, a_tag, pos_match,
                      w_morph, a_morph, morph_match, freq_brwac, freq_cb, genre,
                      sim_ctx, sim_resp_match, sim_resp_ctx,
                      sim_ctx_ft, sim_resp_match_ft, sim_resp_ctx_ft,
                      time_start, time_dig, total_time,
                      w_lemma, a_lemma):

    norm_freq_brwac = 0
    log_freq_brwac = 0
    if freq_brwac > 0:
        norm_freq_brwac = round(freq_brwac * 1000000 / brwac_size, 3)
        log_freq_brwac = round(math.log(norm_freq_brwac)+3, 3)

    norm_freq_bra = 0
    log_freq_bra = 0
    if freq_cb > 0:
        norm_freq_bra = round(freq_cb * 1000000 / bra_size, 3)
        log_freq_bra = round(math.log(norm_freq_bra)+3, 3)

    line = '%s\t%s\t%s\t%s\t%s\t%s\t' \
           '%s\t%s\t%s\t%s\t%s\t%s\t%s\t' \
           '%s\t' \
           '%s\t%s\t%s\t%s\t%s\t' \
           '%s\t%s\t%s\t' \
           '%s\t%s\t%s\t%s\t%s\t' \
           '%s\t%s\t%s\t' \
           '%s\t%s\t%s\t' \
           '%s\t%s\t%s\t' \
           '%s\t%s\n' \
           % (part, word_id, text_id, widx, sent, wsidx,
              word_place_in_sent, sent_length, w_raw, w, len(w), a_raw, a,
              ortho_match,
              tag, content_func, tag_desc, a_tag, pos_match,
              w_morph, a_morph, morph_match,
              norm_freq_brwac, log_freq_brwac, norm_freq_bra, log_freq_bra, genre,
              sim_ctx, sim_resp_match, sim_resp_ctx,
              sim_ctx_ft, sim_resp_match_ft, sim_resp_ctx_ft,
              time_start, time_dig, total_time,
              w_lemma, a_lemma)

    f.write(line)


def get_text_ids_pars_and_genres():
    df_pars_eye = pd.read_csv('data/50pars_eye.txt', sep='\t', escapechar='\\')
    df_pars_cloze = pd.read_csv('data/50pars_cloze.txt', sep='\t', escapechar='\\')

    text_id_refs = {}
    text_id_genres = {}
    for i, p in enumerate(df_pars_cloze['Paragraph']):
        if i < 20:
            text_id_genres[i + 1] = 'DC'
        elif i < 40:
            text_id_genres[i + 1] = 'JN'
        else:
            text_id_genres[i + 1] = 'LT'

        for j, p2 in enumerate(df_pars_eye['Paragraph']):
            if p == p2:
                text_id_refs[i + 1] = j + 1

    return text_id_refs, text_id_genres, df_pars_cloze['Paragraph']

tag_map_palavras = {'N': 'Nome',
                  'PROP': 'Nome',
                  'SPEC': 'SPEC',
                  'DET': 'Artigo',
                  'ART': 'Artigo',
                  'PERS': 'Pronome',
                  'PRON-PERS': 'Pronome',
                  'PRON-DET': 'Pronome',
                  'PRON-INDP': 'Pronome',
                  'ADJ': 'Adjetivo',
                  'ADV': 'Advérbio',
                  'V': 'Verbo',
                  'V-FIN': 'Verbo',
                  'V-INF': 'Verbo',
                  'V-PCP': 'Verbo',
                  'V-GER': 'Verbo',
                  'VAUX': 'Verbo',
                  'VAUX-S': 'Verbo',
                  'IMP': 'Verbo',
                  'NUM': 'Numeral',
                  'PRP': 'Preposição',
                  'KS': 'Conjunção',
                  'KC': 'Conjunção',
                  'CONJ-S': 'Conjunção',
                  'CONJ-C': 'Conjunção',
                  'IN': 'Interjeição',
                  'INTJ': 'Interjeição',
                  'EC': 'Elemento Composto',
                  'PU': "Pontuação",
                  'ERR': 'Erro'}


def content_or_function_word(tag):
    if tag in ['N', 'PROP', 'V', 'ADJ', 'ADV', 'V-FIN', 'V-INF', 'V-PCP', 'V-GER', 'VAUX', 'VAUX-S', 'IMP']:
        return 'Content'
    else:
        return 'Function'


def get_pos_word_ctl(ctl, w):
    if w not in ctl:
        ctl[w] = 1
    else:
        ctl[w] += 1

    return '%s_%s' % (w, ctl[w])


def get_sentence_lengths(df):
    sentence_lengths = {}
    sentences = {}
    for i, p in enumerate(df['Parágrafo']):
        sent_len_key = '%s_%s' % (p, df['Sentença'][i])
        if sent_len_key not in sentence_lengths:
            max_wid = 0
            last_wid = 0
            sentences[sent_len_key] = {}
            for j, wid in enumerate(df['Índice Palavra']):
                if wid == last_wid:
                    continue
                if df['Parágrafo'][j] == p and df['Sentença'][j] == df['Sentença'][i]:
                    sentences[sent_len_key][wid] = df['Palavra'][j]
                    max_wid += 1
                else:
                    if max_wid > 0:
                        sentence_lengths[sent_len_key] = max_wid
                        break
                last_wid = wid
            sentence_lengths[sent_len_key] = max_wid
    return sentence_lengths, sentences


def get_word_place_in_sent(wsidx, sent_length):
    if wsidx > (sent_length * 0.75):
        word_place_in_sent = 4
    elif wsidx > (sent_length * 0.5):
        word_place_in_sent = 3
    elif wsidx > (sent_length * 0.25):
        word_place_in_sent = 2
    else:
        word_place_in_sent = 1

    return word_place_in_sent


brwac_size = 2691373903

df_freq_brwac = pd.read_csv('data/lista_brWaC_geral_v3_nlpnet.tsv', sep='\t')
cache_freq_brwac = {}


def get_freq_brwac(dff, w, tag):
    freq = 0
    key = "%s" % w
    # key = "%s_%s" % (w, tag.lower())
    try:
        if key in cache_freq_brwac:
            return cache_freq_brwac[key]

        df_tmp = dff.loc[dff['Palavra'].str.lower() == w]
        freq = df_tmp['Frequência'].sum()
        # df_tmp = df_tmp.loc[df_tmp['Tag'].str.lower() == tag.lower()]
        # freq = df_tmp.iloc[0]['Frequência']
    except Exception as exc:
        print("freq-brwac", w, tag, "-->", exc)

    cache_freq_brwac[key] = freq
    return freq


bra_size = 654481742

df_freq_bra = pd.read_csv('data/freq_nathan.csv',
                          dtype={"word": "string", "corpus_joao-rodrigues_processed.txt": int, "chc.txt": int,
                                 "corpus_nilc.txt": int, "fapesp.txt": int, "folhinha.txt": int, "g1.txt": int,
                                 "googlenews.txt": int, "lacioweb.txt": int, "livros-dominio-publico.txt": int,
                                 "livros_didaticos.txt": int, "livros_portugues.txt": int, "mundo_estranho.txt": int,
                                 "para_seu_filho_ler.txt": int, "plnbr.txt": int, "saresp.txt": int,
                                 "SubIMDB_pt.txt": int, "wiki20-10-2016.txt": int, "corpus Nathan": int,
                                 "corpus Nathan2": int, "corpus_brasileiro.txt": int})

cache_freq_bra = {}


def get_freq_cb(dff, w):
    freq = 0
    try:
        if w in cache_freq_bra:
            return cache_freq_bra[w]

        freq = dff.loc[dff['word'] == w].iloc[0]['corpus_brasileiro.txt']
    except Exception as exc:
        print("freq-CB", w, "-->", exc)

    cache_freq_bra[w] = freq
    return freq


# semantic similarity =================

def getVectorsEmbeddings(text, embeddings):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    tokens = text.split()
    # print(tokens)
    dim = embeddings['word'].size
    word_vec = []
    for word in tokens:
        if word.lower() in embeddings:
            word_vec.append(embeddings[word.lower()])
        else:
            word_vec.append(np.random.uniform(-0.01, 0.01, dim))

    if len(word_vec) == 0:
        word_vec.append(np.random.uniform(-0.01, 0.01, dim))

    return word_vec


def similarity(vec, context_vec):
    context_vec_avg = np.average(context_vec, axis=0)
    return 1 - spatial.distance.cosine(vec, context_vec_avg)


def get_similarity(word, text):

    tokens = word.split(" ")
    if len(tokens) > 1:
        word = tokens[0]

    ft_word_emb = getVectorsEmbeddings(word, embeddings['fasttext'])
    ft_text_emb = getVectorsEmbeddings(text, embeddings['fasttext'])
    try:
        ft_sim = similarity(ft_word_emb, ft_text_emb)
    except Exception as exc:
        print("===========================================================================================")
        print("---", word)
        print("---", text)
        print("---", ft_word_emb)
        print("---", ft_text_emb)
        print(exc)
        print("===========================================================================================")
        exit(1)

    if ft_sim > 1:
        ft_sim = 1
    if ft_sim < 0:
        ft_sim = 0

    context = '[CLS] %s [MASK] [SEP]' % text
    bert_score = calc_score_task1(context, word, tokenizer, model)
    if bert_score is None:
        bert_score = -100
    # ret = bert_score
    # if ret is None:
    #     print("bert - none: ", word, ft_sim)
    #     ret = ft_sim

    return bert_score, ft_sim


def get_similarity_match(word, answer, text):
    tokens = answer.split(" ")
    if len(tokens) > 1:
        answer = tokens[0]

    ft_word_emb = getVectorsEmbeddings(word, embeddings['fasttext'])
    ft_text_emb = getVectorsEmbeddings(answer, embeddings['fasttext'])
    ft_sim = similarity(ft_word_emb, ft_text_emb)

    if ft_sim > 1:
        ft_sim = 1
    if ft_sim < 0:
        ft_sim = 0

    context = '[CLS] %s [MASK] [SEP]' % text
    bert_score = calc_score_task2(context, word, answer, tokenizer, model)
    if bert_score is None:
        bert_score = -100
    # ret = bert_score
    # if ret is None:
    #     print("bert - none: ", word, answer, ft_sim)
    #     ret = ft_sim

    return bert_score, ft_sim


embeddings = {}


def load_embeddings():
    fasttext_embedding_file = 'data/bbp_fasttext_cbow_300d.txt'
    embeddings['fasttext'] = gensim.models.KeyedVectors.load_word2vec_format(fasttext_embedding_file)


# semantic similarity =====================


def anonymize_participants(df):
    participants = {}

    fp = open("out/participants_%s.tsv" % process_date, "w")
    fp.write("ID\tParticipant\tEmail\tIdade\n")

    part_count = 1
    for i, p in enumerate(df['Nome Participante']):
        if p not in participants:
            participants[p] = part_count
            email = df['Email'][i]
            idade = df['Idade'][i]
            fp.write("%s\t%s\t%s\t%s\n" % (part_count, p, email, idade))
            part_count += 1

    fp.close()

    return participants


def load_answers_annotations():
    answers_annotation = pd.read_csv('data/answers_annotation.tsv', sep='\t')
    answers_annotation = answers_annotation.fillna('')
    return answers_annotation


annotation_cache = {}


def get_annotation(answer):
    correcao, tag, morph = '', '', ''

    if answer.strip() == "":
        return "", "RANDOM", ""

    if answer in annotation_cache:
        correcao = annotation_cache[answer]['correcao']
        tag = annotation_cache[answer]['tag']
        morph = annotation_cache[answer]['morph']
        return correcao, tag, morph

    try:
        itens = answers_annotation.loc[answers_annotation['Resposta'] == answer]
        first = itens.iloc[0]
        correcao = first['Correção']
        tag = first['PoS']
        morph = first['Morph']

    except Exception as exc:
        # print("get_annotation", answer, "-->", exc)
        correcao, tag, morph = '', '', ''

    annotation_cache[answer] = {}
    annotation_cache[answer]['correcao'] = correcao
    annotation_cache[answer]['tag'] = tag
    annotation_cache[answer]['morph'] = morph

    return correcao, tag, morph


def call_palavras(text):
    url = "http://fw.nilc.icmc.usp.br:23380/api/v1/palavras/tigerxml/m3tr1x01"
    try:
        text = text.replace('"', '').replace('”', '')
        # ret = requests.post(url, {"content": text, "options": "--dep-fuse"}).text
        ret = requests.post(url, {"content": text, "options": "--dep-retokenize"}).text

        ret = ret.replace("\n", "").replace("\t", "").replace("</ß></ß>", "</ß>")
        print(ret)
        doc = xml.dom.minidom.parseString(ret)
        return doc
    except Exception as exc:
        print("palavras-online", text, "-->", exc)


def get_pos_palavras_answer(preceding_passage, a):
    ret = {}

    try:
        new_text = "%s %s" % (preceding_passage, a)
        doc = call_palavras(new_text)
        ts = doc.getElementsByTagName('t')

        last_t = ts[len(ts)-1]

        ret['pos'] = last_t.getAttribute("pos").upper()
        ret['lemma'] = last_t.getAttribute("lemma")
        ret['morph'] = ""
        if last_t.getAttribute("morph") != "--":
            ret['morph'] = last_t.getAttribute("morph")

    except Exception as exc:
        print("get_pos_palavras_answer", preceding_passage, a, "-->", exc)

    return ret


def get_pos_palavras(text):
    pos_words_ctl = {}
    dict_ret = {}
    try:
        doc = call_palavras(text)
        ts = doc.getElementsByTagName('t')
        for t in ts:
            word_tag = t.getAttribute("word")
            if "_" in word_tag:
                for w_item in word_tag.split('_'):
                    word_tag = get_pos_word_ctl(pos_words_ctl, w_item.lower())
                    dict_ret[word_tag] = {}
                    dict_ret[word_tag]['pos'] = t.getAttribute("pos").upper()
                    dict_ret[word_tag]['lemma'] = t.getAttribute("lemma")
                    dict_ret[word_tag]['morph'] = ""
                    if t.getAttribute("morph") != "--":
                        dict_ret[word_tag]['morph'] = t.getAttribute("morph")
            else:
                word_tag = get_pos_word_ctl(pos_words_ctl, word_tag.lower())
                dict_ret[word_tag] = {}
                dict_ret[word_tag]['pos'] = t.getAttribute("pos").upper()
                dict_ret[word_tag]['lemma'] = t.getAttribute("lemma")
                dict_ret[word_tag]['morph'] = ""
                if t.getAttribute("morph") != "--":
                    dict_ret[word_tag]['morph'] = t.getAttribute("morph")

    except Exception as exc:
        print("palavras-online", text, "-->", exc)

    return dict_ret


cache_tags_pos_palavras_words = {
    "UID_9_50_01": {"pos": "NUM", "lemma": "01", "morph": ""},
    "UID_13_64_02": {"pos": "NUM", "lemma": "02", "morph": ""},
    "UID_9_19_005": {"pos": "NUM", "lemma": "005", "morph": ""},
    "UID_32_36_19": {"pos": "NUM", "lemma": "19", "morph": ""},
    "UID_13_51_32": {"pos": "NUM", "lemma": "32", "morph": ""},
    "UID_18_11_343": {"pos": "NUM", "lemma": "343", "morph": ""},
    "UID_48_13_d": {"pos": "PROP", "lemma": "d", "morph": ""},
    "UID_15_46_do": {"pos": "PRP", "lemma": "de+o", "morph": ""},
    "UID_23_52_do": {"pos": "PRP", "lemma": "de+o", "morph": ""},
    "UID_32_12_no": {"pos": "PRP", "lemma": "em+o", "morph": ""},
    "UID_9_36_nos": {"pos": "PRP", "lemma": "em+o", "morph": ""},
    "UID_46_26_nisso": {"pos": "PRP", "lemma": "em+isso", "morph": ""},
    "UID_2_31_dentre": {"pos": "PRP", "lemma": "de+entre", "morph": ""},
    "UID_4_60_repeti-la": {"pos": "V-INF", "lemma": "repetir", "morph": "1S"},
    "UID_8_34_depositando-se": {"pos": "V-GER", "lemma": "depositar", "morph": ""},
    "UID_5_8_torna-se": {"pos": "V-FIN", "lemma": "tornar", "morph": "PR 3S IND VFIN"},
    "UID_27_45_tornou-se": {"pos": "V-FIN", "lemma": "tornar", "morph": "PS 3S IND VFIN"},
    "UID_44_23_d’água": {"pos": "PRP", "lemma": "de", "morph": ""},
    "UID_44_34_embebendo-se": {"pos": "V-GER", "lemma": "embeber", "morph": ""},
    "UID_44_35_embebendo-se": {"pos": "V-GER", "lemma": "embeber", "morph": ""},
    "UID_46_35_resolveu-se": {"pos": "V-FIN", "lemma": "resolver", "morph": "PS 3S IND VFIN"},
    "UID_46_41_aparecer-lhe": {"pos": "V-INF", "lemma": "aparecer", "morph": "3S"}
}


def get_pos_palavras_cache(tags_pos_palavras, word_id, w, wtg):
    p_tag, p_lemma, p_morph = "", "", ""
    try:
        word_id_palavras_key = "%s_%s" % (word_id, w)
        if word_id_palavras_key in cache_tags_pos_palavras_words:
            p_tag = cache_tags_pos_palavras_words[word_id_palavras_key]["pos"]
            p_lemma = cache_tags_pos_palavras_words[word_id_palavras_key]["lemma"]
            p_morph = cache_tags_pos_palavras_words[word_id_palavras_key]["morph"]
        else:
            p_tag = tags_pos_palavras[wtg]["pos"]
            p_lemma = tags_pos_palavras[wtg]["lemma"]
            p_morph = tags_pos_palavras[wtg]["morph"]

            cache_tags_pos_palavras_words[word_id_palavras_key] = {}
            cache_tags_pos_palavras_words[word_id_palavras_key]["pos"] = p_tag
            cache_tags_pos_palavras_words[word_id_palavras_key]["lemma"] = p_lemma
            cache_tags_pos_palavras_words[word_id_palavras_key]["morph"] = p_morph

    except Exception as exc:
        print("palavras-online", word_id, w, "-->", exc)
        p_tag = "ERR"

    return p_tag, p_lemma, p_morph



def get_pos_palavras_answer_cache(preceding_passage, word_id, a):
    p_tag, p_lemma, p_morph = "", "", ""
    try:
        word_id_palavras_key = "%s_%s" % (word_id, clean_word(a))
        if word_id_palavras_key in cache_tags_pos_palavras_words:
            p_tag = cache_tags_pos_palavras_words[word_id_palavras_key]["pos"]
            p_lemma = cache_tags_pos_palavras_words[word_id_palavras_key]["lemma"]
            p_morph = cache_tags_pos_palavras_words[word_id_palavras_key]["morph"]
        else:
            pos_answer = get_pos_palavras_answer(preceding_passage, a)
            p_tag = pos_answer["pos"]
            p_lemma = pos_answer["lemma"]
            p_morph = pos_answer["morph"]

            cache_tags_pos_palavras_words[word_id_palavras_key] = {}
            cache_tags_pos_palavras_words[word_id_palavras_key]["pos"] = p_tag
            cache_tags_pos_palavras_words[word_id_palavras_key]["lemma"] = p_lemma
            cache_tags_pos_palavras_words[word_id_palavras_key]["morph"] = p_morph

    except Exception as exc:
        print("get_pos_palavras_answer_cache", word_id, a, "-->", exc)
        p_tag = "ERR"

    return p_tag, p_lemma, p_morph


#BEGIN BERT
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

def load_bert_model():
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased')

    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained('neuralmind/bert-large-portuguese-cased')
    model.eval()

    return tokenizer, model


def calc_score_task1(text, target_word, tokenizer, model, debug=False):
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # get indexs
    target_index = tokenizer.convert_tokens_to_ids([target_word])[0]
    masked_index = tokenized_text.index('[MASK]')

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # get words again
    expected_token = tokenizer.convert_ids_to_tokens([target_index])[0]

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    # normalise between 0 and 1
    # predictions_candidates = torch.sigmoid(predictions[0][0][masked_index])
    predictions_candidates = predictions[0][0][masked_index]

    predicted_index = torch.argmax(predictions_candidates).item()

    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

    # if word dont exist return 0
    if expected_token == '[UNK]':
        return None

    predictions_candidates = predictions_candidates.cpu().numpy()
    target_bert_confiance = predictions_candidates[target_index]
    predicted_bert_confience = predictions_candidates[predicted_index]

    #score = 1 - (target_bert_confiance - predicted_bert_confience if target_bert_confiance > predicted_bert_confience else predicted_bert_confience - target_bert_confiance)
    score = target_bert_confiance - predicted_bert_confience if target_bert_confiance > predicted_bert_confience else predicted_bert_confience - target_bert_confiance
    if debug:
        print("predicted token ---> ", predicted_token, predicted_bert_confience)
        print("expected token  ---> ", expected_token, target_bert_confiance)
        print("Score:", score)

    return score


def calc_score_task2(text, target_word, predicted_word, tokenizer, model, debug=False):

  tokenized_text = tokenizer.tokenize(text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

  #get indexs
  target_index = tokenizer.convert_tokens_to_ids([target_word])[0]
  predicted_index = tokenizer.convert_tokens_to_ids([predicted_word])[0]
  masked_index = tokenized_text.index('[MASK]')

  # Create the segments tensors.
  segments_ids = [0] * len(tokenized_text)

  # Convert inputs to PyTorch tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])


  # get words again
  expected_token = tokenizer.convert_ids_to_tokens([target_index])[0]
  predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]


  # Predict all tokens
  with torch.no_grad():
      predictions = model(tokens_tensor, segments_tensors)

  # normalise between 0 and 1
  # predictions_candidates = torch.sigmoid(predictions[0][0][masked_index]).cpu().numpy()
  predictions_candidates = predictions[0][0][masked_index].cpu().numpy()

  # get words again
  expected_token = tokenizer.convert_ids_to_tokens([target_index])[0]
  predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

  # if word dont exist return 0
  if predicted_token == '[UNK]' or expected_token == '[UNK]':
    return None

  target_bert_confiance = predictions_candidates[target_index]
  predicted_bert_confience = predictions_candidates[predicted_index]


  #score = 1 - (target_bert_confiance-predicted_bert_confience if  target_bert_confiance > predicted_bert_confience else predicted_bert_confience-target_bert_confiance)
  score = target_bert_confiance - predicted_bert_confience if target_bert_confiance > predicted_bert_confience else predicted_bert_confience - target_bert_confiance

  if debug:
    print("predicted token ---> ", predicted_token, predicted_bert_confience)
    print("expected token  ---> ",expected_token , target_bert_confiance)
    print("Score:", score)

  return score

#END BERT



def process(thread_name, df):

    total_lines_count = 1
    wsidx, sent_length, word_place_in_sent = 0, 0, 0
    last_sent, last_text = 0, 0
    pos_words_ctl, cache_sim, cache_sim_ctx = {}, {}, {}
    preceding_passage, last_word_id, last_word = "", "", ""
    wtg = 0
    sentence, tags_pos_palavras, cache_tags_pos_palavras_sents = {}, {}, {}
    text_sent, first_word = "", ""
    first_word_of_a_text = False

    f = open("out/%s_%s" % (thread_name, output_file), "w")
    write_output_header(f)

    total_lines = df['Palavra'].count()

    for i, w in enumerate(df['Palavra']):
        cloze_par_id = df['Parágrafo'][i]
        w_raw = df['Palavra Crua'][i]
        text_id = text_id_refs[cloze_par_id]
        text_par = paragraphs[cloze_par_id - 1]  # texto do parágrafo
        widx = df['Índice Palavra'][i]

        word_id = "UID_%s_%s" % (text_id, widx)

        part_name = df['Nome Participante'][i]
        part = participants[part_name]

        sent_idx = df['Sentença'][i]
        sent_str_id = '%s_%s' % (cloze_par_id, sent_idx)

        if word_id != last_word_id:
            first_word_of_a_text = False
            preceding_passage = '%s %s' % (preceding_passage, last_word)
            if sent_idx == last_sent and text_id == last_text:
                wsidx += 1
                word_place_in_sent = get_word_place_in_sent(wsidx, sent_length)
            else:
                wsidx = 1
                word_place_in_sent = 1
                sent_length = sentence_lengths[sent_str_id]
                sentence = sentences[sent_str_id]

        if text_id != last_text:
            first_word_of_a_text = True
            print(text_par)
            first_word = text_par.split(' ')[0]
            preceding_passage = first_word
            last_sent = 0
            wsidx = 2

        if sent_idx != last_sent:
            text_sent = ""
            pos_words_ctl = {}
            if sent_idx == 1:
                text_sent +=  "%s " % first_word
            print(sentence)

            for k, v in sentence.items():
                text_sent += "%s " % v

            print("-------", text_sent)
            if sent_str_id in cache_tags_pos_palavras_sents:
                tags_pos_palavras = cache_tags_pos_palavras_sents[sent_str_id]
            else:
                tags_pos_palavras = get_pos_palavras(text_sent)
                cache_tags_pos_palavras_sents[sent_str_id] = tags_pos_palavras

            print(tags_pos_palavras)

        genre = text_id_genres[cloze_par_id]
        time_start = df['Tempo Início(ms)'][i]
        time_dig = df['Tempo Digitação(ms)'][i]
        total_time = time_start + time_dig

        a = df['Resposta'][i]
        a_raw = a
        w = clean_word(w)
        a = clean_word(a)

        if word_id != last_word_id:
            wtg = get_pos_word_ctl(pos_words_ctl, w)

        tag = "ERR"
        tag, w_lemma, w_morph = get_pos_palavras_cache(tags_pos_palavras, word_id, w, wtg)

        tag_desc = tag_map_palavras[tag]
        content_func = content_or_function_word(tag)

        print(thread_name, round((time.time() - ini)/60, 2), total_lines_count, "/", total_lines,
              ">", text_id, sent_idx, widx, word_id, w, tag, w_morph, a)


        last_sent = sent_idx
        last_text = text_id
        last_word_id = word_id
        last_word = w

        # PoS
        a_lemma, a_tag, a_morph = '', '', ''
        a_correcao, a_tag, a_morph = get_annotation(a)
        if a_correcao != '':
            a = a_correcao
        if a_tag != '':
            a_lemma, a_tag, a_morph = a, a_tag, a_morph

        if a_tag == '':
            a_tag, a_lemma, a_morph = get_pos_palavras_answer_cache(preceding_passage, word_id, a)

        ortho_match = 0
        if w == a.lower():
            ortho_match = 1

        pos_match = 0
        if tag == a_tag:
            pos_match = 1

        morph_match = 0
        if w_morph == a_morph:
            morph_match = 1


        # contextual_fit / LSA_Context_Score
        if word_id in cache_sim_ctx:
            sim_ctx_bert, sim_ctx_ft = cache_sim_ctx[word_id][0], cache_sim_ctx[word_id][1]
        else:
            sim_ctx_bert, sim_ctx_ft = get_similarity(w, preceding_passage)
            cache_sim_ctx[word_id] = [sim_ctx_bert, sim_ctx_ft]

        # semantic_relatedness / LSA_Response_Match_Score
        cache_sim_key = "%s_%s" % (word_id, a)
        if cache_sim_key in cache_sim:
            sim_resp_match_bert, sim_resp_ctx_bert = cache_sim[cache_sim_key][0], cache_sim[cache_sim_key][1]
            sim_resp_match_ft, sim_resp_ctx_ft = cache_sim[cache_sim_key][2], cache_sim[cache_sim_key][3]
        else:
            sim_resp_ctx_bert, sim_resp_ctx_ft = get_similarity(a, preceding_passage)
            sim_resp_match_bert, sim_resp_match_ft = get_similarity_match(w, a, preceding_passage)
            cache_sim[cache_sim_key] = [sim_resp_match_bert, sim_resp_ctx_bert, sim_resp_match_ft, sim_resp_ctx_ft]

        freq_cb = get_freq_cb(df_freq_bra, w)
        freq_brwac = get_freq_brwac(df_freq_brwac, w, tag)

        if first_word_of_a_text and widx > 1:
            write_output_line(f, part, "UID_%s_1" % text_id, text_id, widx-1, sent_idx, wsidx-1, word_place_in_sent,
                              sent_length, first_word,
                              clean_word(first_word), '', '', 0, '', '', '', '', 0,
                              '', '', 0, 0, 0, genre,
                              0, 0, 0,
                              0, 0, 0,
                              0, 0, 0,
                              '', '')

        write_output_line(f, part, word_id, text_id, widx, sent_idx, wsidx, word_place_in_sent, sent_length, w_raw,
                          w, a_raw, a, ortho_match, tag, content_func, tag_desc, a_tag, pos_match,
                          w_morph, a_morph, morph_match, freq_brwac, freq_cb, genre,
                          sim_ctx_bert, sim_resp_match_bert, sim_resp_ctx_bert,
                          sim_ctx_ft, sim_resp_match_ft, sim_resp_ctx_ft,
                          time_start, time_dig, total_time,
                          w_lemma, a_lemma)

        total_lines_count += 1

    print("Total lines:", total_lines_count)

    print("Tempo Exec: ", (time.time() - ini)/60)

    f.close()



if __name__ == '__main__':

    print('--- Inicia processamento...')
    ini = time.time()

    print('--- Carrega o dataset principal...')
    main_df = load_main_dataset()
    print("Tempo: ", time.time() - ini)

    print('--- anonimização participantes...')
    participants = anonymize_participants(main_df)
    print("Tempo: ", time.time() - ini)

    print('--- Carregando embeddings...')
    load_embeddings()
    print("Tempo: ", time.time() - ini)

    print('--- Carregando answears annotation...')
    answers_annotation = load_answers_annotations()
    print("Tempo: ", time.time() - ini)

    text_id_refs, text_id_genres, paragraphs = get_text_ids_pars_and_genres()

    print('--- calcula tamanhos das sentenças...')
    sentence_lengths, sentences = get_sentence_lengths(main_df)
    print("Tempo: ", time.time() - ini)

    print('--- carrega bert...')
    tokenizer, model = load_bert_model()
    print("Tempo: ", time.time() - ini)

    print(sentences)

    print('--- Inicia loop das palavras...')

    #dividir
    pars_to_process_1 = list(range(1,11))   # 01 02 03 04 05 06 07 08 09 10
    pars_to_process_2 = list(range(11,21))  # 11 12 13 14 15 16 17 18 19 20
    pars_to_process_3 = list(range(21,31))  # 21 22 23 24 25 26 27 28 29 30
    pars_to_process_4 = list(range(31,41))  # 31 32 33 34 35 36 37 38 39 40
    pars_to_process_5 = list(range(41,51))  # 41 42 43 44 45 46 47 48 49 50
    df1 = main_df[main_df['Parágrafo'].isin(pars_to_process_1)].copy()
    df2 = main_df[main_df['Parágrafo'].isin(pars_to_process_2)].copy()
    df3 = main_df[main_df['Parágrafo'].isin(pars_to_process_3)].copy()
    df4 = main_df[main_df['Parágrafo'].isin(pars_to_process_4)].copy()
    df5 = main_df[main_df['Parágrafo'].isin(pars_to_process_5)].copy()

    df1 = df1.reset_index()
    df2 = df2.reset_index()
    df3 = df3.reset_index()
    df4 = df4.reset_index()
    df5 = df5.reset_index()

    # process("t1", df1)
    t1 = threading.Thread(target=process, args=('t1', df1))
    t2 = threading.Thread(target=process, args=('t2', df2))
    t3 = threading.Thread(target=process, args=('t3', df3))
    t4 = threading.Thread(target=process, args=('t4', df4))
    t5 = threading.Thread(target=process, args=('t5', df5))

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()

    out1 = pd.read_csv("out/%s_%s" % ('t1', output_file), sep='\t', quoting=3)
    out2 = pd.read_csv("out/%s_%s" % ('t2', output_file), sep='\t', quoting=3)
    out3 = pd.read_csv("out/%s_%s" % ('t3', output_file), sep='\t', quoting=3)
    out4 = pd.read_csv("out/%s_%s" % ('t4', output_file), sep='\t', quoting=3)
    out5 = pd.read_csv("out/%s_%s" % ('t5', output_file), sep='\t', quoting=3)

    out = out1.append(out2, ignore_index=True)
    out = out.append(out3, ignore_index=True)
    out = out.append(out4, ignore_index=True)
    out = out.append(out5, ignore_index=True)

    out['Semantic_Word_Context_Score'] = round(1 - (
            out['Semantic_Word_Context_Score'] / out['Semantic_Word_Context_Score'].max()), 3)
    out['Semantic_Response_Match_Score'] = round(1 - (
            out['Semantic_Response_Match_Score'] / out['Semantic_Response_Match_Score'].max()), 3)
    out['Semantic_Response_Context_Score'] = round(1 - (
            out['Semantic_Response_Context_Score'] / out['Semantic_Response_Context_Score'].max()), 3)

    out.loc[out['Semantic_Word_Context_Score'] > 1, "Semantic_Word_Context_Score"] \
        = round(out['Semantic_Word_Context_Score_ft'], 3)
    out.loc[out['Semantic_Response_Match_Score'] > 1, "Semantic_Response_Match_Score"] \
        = round(out['Semantic_Response_Match_Score_ft'], 3)
    out.loc[out['Semantic_Response_Context_Score'] > 1, "Semantic_Response_Context_Score"] \
        = round(out['Semantic_Response_Context_Score_ft'], 3)

    out = out.drop('Semantic_Word_Context_Score_ft', axis='columns')
    out = out.drop('Semantic_Response_Match_Score_ft', axis='columns')
    out = out.drop('Semantic_Response_Context_Score_ft', axis='columns')

    out = out.sort_values(['Text_ID', 'Participant', 'Word_Number'])
    out.to_csv("out/%s" % output_file, sep='\t', quoting=3, index=False)
