import math
import time
import pandas as pd
import nlpnet
import numpy as np
import string
import pickle
from scipy import spatial
import gensim

output_file = "cloze_predict_29_FULL.tsv"
process_date = "2020_10_25"


def clean_word(word):
    w = "%s" % word
    w = w.lower()
    w = w.replace('”', '')
    w = w.replace('“', '')
    w = w.replace('"', '')
    w = w.replace('...', '')
    w = w.replace('%', '')
    w = w.replace('?', '')
    w = w.replace('.', '')
    w = w.replace(']', '')
    w = w.replace('[', '')
    w = w.replace('\\', '')
    w = w.replace(' ', '')
    w = w.replace(' ', '_')  # \xa0
    w = w.strip()
    w = w.strip(',')
    return w


def load_main_dataset():
    df_puc = pd.read_csv('data/cloze_puc_%s.csv' % process_date)
    df_usp = pd.read_csv('data/cloze_usp_%s.csv' % process_date)
    df_ufc = pd.read_csv('data/cloze_ufc_%s.csv' % process_date)
    df_ufabc = pd.read_csv('data/cloze_ufabc_%s.csv' % process_date)
    df_utfpr = pd.read_csv('data/cloze_utfpr_%s.csv' % process_date)
    df_uerj = pd.read_csv('data/cloze_uerj_%s.csv' % process_date)

    ret = df_puc.append(df_usp, ignore_index=True)
    ret = ret.append(df_ufc, ignore_index=True)
    ret = ret.append(df_ufabc, ignore_index=True)
    ret = ret.append(df_utfpr, ignore_index=True)
    ret = ret.append(df_uerj, ignore_index=True)

    print("puc:", df_puc['Palavra'].count())
    print("usp:", df_usp['Palavra'].count())
    print("ufc:", df_ufc['Palavra'].count())
    print("ufabc:", df_ufabc['Palavra'].count())
    print("utfpr:", df_utfpr['Palavra'].count())
    print("uerj:", df_uerj['Palavra'].count())
    print("total:", ret['Palavra'].count())

    return ret


def write_output_header():
    header = 'Participant\tWord_Unique_ID\tText_ID\tWord_Number\tSentence_Number\tWord_In_Sentence_Number\t' \
             'Word_Place_In_Sent\tSent_Length\tWord\tWord_Cleaned\tWord_Length\tAnswer\tOrthographicMatch\t' \
             'POS\tWord_Content_Or_Function\tWord_POS\tWord_Tag_DELAF\tAnswer_Tag_DELAF\tPOSMatch\t' \
             'Word_Morph_DELAF\tAnswer_Morph_DELAF\tInflectionMatch\tFreq_Corpus_Brasileiro\tLog_Freq_Brasileiro\t' \
             'Freq_brWaC\tLog_Freq_brWaC\tGenre\t' \
             'LSA_Context_Score\tLSA_Response_Match_Score\tFastText_Context_Score\tFastText_Response_Match_Score\t' \
             'Word2Vec_Context_Score\tWord2Vec_Response_Match_Score\tAvg_Context_Score\tAvg_Response_Match_Score\t' \
             'Time_to_Start\tTyping_Time\tTotal_time\n'
    print(header)
    f.write(header)


def write_output_line(part, word_id, text_id, widx, sent, wsidx, word_place_in_sent, sent_length, w_raw, w, a,
                      ortho_match, tag, content_func, tag_desc, w_tag_delaf, a_tag_delaf, pos_match,
                      w_morph_delaf, a_morph_delaf, morph_match, freq_cb, freq_brwac, genre, lsa_sim_ctx,
                      lsa_sim, ft_sim_ctx, ft_sim, w2v_sim_ctx, w2v_sim, time_start, time_dig, total_time):

    avg_sim_ctx = round((lsa_sim_ctx + ft_sim_ctx + w2v_sim_ctx)/3, 5)
    avg_sim = round((lsa_sim + ft_sim + w2v_sim)/3, 5)

    log_freq_cb, log_freq_brwac = 0, 0
    if freq_cb > 0:
        log_freq_cb = round(math.log(freq_cb), 5)
    if freq_brwac > 0:
        log_freq_brwac = round(math.log(freq_brwac), 5)

    line = '%s\t%s\t%s\t%s\t%s\t%s\t' \
           '%s\t%s\t%s\t%s\t%s\t%s\t%s\t' \
           '%s\t%s\t%s\t%s\t%s\t%s\t' \
           '%s\t%s\t%s\t%s\t%s\t' \
           '%s\t%s\t%s\t' \
           '%s\t%s\t%s\t%s\t' \
           '%s\t%s\t%s\t%s\t' \
           '%s\t%s\t%s\n' \
           % (part, word_id, text_id, widx, sent, wsidx,
              word_place_in_sent, sent_length, w_raw, w, len(w), a, ortho_match,
              tag, content_func, tag_desc, w_tag_delaf, a_tag_delaf, pos_match,
              w_morph_delaf, a_morph_delaf, morph_match, freq_cb, log_freq_cb,
              freq_brwac, log_freq_brwac, genre,
              lsa_sim_ctx, lsa_sim, ft_sim_ctx, ft_sim,
              w2v_sim_ctx, w2v_sim, avg_sim_ctx, avg_sim,
              time_start, time_dig, total_time)

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


tag_map_nlpnet = {'ADJ': 'Adjetivo',
                  'ADV': 'Advérbio',
                  'ADV-KS': 'Advérbio',
                  'ADV-KS-REL': 'Advérbio',
                  'ART': 'Artigo',
                  'KC': 'Conjunção',
                  'KS': 'Conjunção',
                  'IN': 'Interjeição',
                  'V': 'Verbo',
                  'VAUX': 'Verbo',
                  'N': 'Nome',
                  'NPROP': 'Nome',
                  'NUM': 'Numeral',
                  'PREP': 'Preposição',
                  'PCP': 'Particípio',
                  'PROADJ': 'Pronome',
                  'PRO-KS': 'Pronome',
                  'PROPESS': 'Pronome',
                  'PRO-KS-REL': 'Pronome',
                  'PROSUB': 'Pronome',
                  'PDEN': 'Palavra Denotativa',
                  'PREP+ART': 'Preposição',
                  'PREP+PROADJ': 'Preposição',
                  'PREP+PROSUB': 'Preposição',
                  'PREP+PROPESS': 'Preposição',
                  'ERR': 'Erro'}


def get_pos_nlpnet_word_ctl(ctl, w):
    if w not in ctl:
        ctl[w] = 1
    else:
        ctl[w] += 1

    return '%s_%s' % (w, ctl[w])


def get_pos_nlpnet_all_pars(paragraphs):
    nlpnet.set_data_dir('/home/sidleal/sid/usp/nlpnet/pos-pt')
    tagger = nlpnet.POSTagger()

    pars_tags = {}
    p_id = 0
    for p in paragraphs:
        tags = {}
        pos_words_ctl = {}
        r_tag = tagger.tag(p)
        for s_tag in r_tag:
            for w_tag in s_tag:
                word_tag = w_tag[0].lower()
                word_tag = word_tag.replace(',', '.')
                word_tag = get_pos_nlpnet_word_ctl(pos_words_ctl, word_tag)

                tags[word_tag] = w_tag[1]
                if '-' in word_tag:
                    w1 = get_pos_nlpnet_word_ctl(pos_words_ctl, word_tag.split('-')[0])
                    tags[w1] = w_tag[1]
                    w2 = get_pos_nlpnet_word_ctl(pos_words_ctl, word_tag.split('-')[1])
                    tags[w2] = w_tag[1]
                # print(p_id, word_tag, w_tag[1])
        pars_tags[p_id] = tags
        p_id += 1

    return pars_tags


def content_or_function_word(tag):
    if tag in ['N', 'NPROP', 'V', 'ADJ', 'ADV', 'ADV-KS', 'ADV-KS-REL', 'VAUX', 'PCP']:
        return 'Content'
    else:
        return 'Function'


def get_sentence_lengths():
    sentence_lengths = {}
    for i, p in enumerate(df['Parágrafo']):
        sent_len_key = '%s_%s' % (p, df['Sentença'][i])
        if sent_len_key not in sentence_lengths:
            max_wid = 0
            for j, wid in enumerate(df['Índice Palavra']):
                if df['Parágrafo'][j] == p and df['Sentença'][j] == df['Sentença'][i]:
                    max_wid += 1
                else:
                    if max_wid > 0:
                        sentence_lengths[sent_len_key] = max_wid
                        break
    return sentence_lengths


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


df_freq_brwac = pd.read_csv('data/lista_brWaC_geral_v3_nlpnet.tsv', sep='\t')
cache_freq_brwac = {}


def get_freq_brwac(dff, w, tag):
    freq = 0
    key = "%s_%s" % (w, tag.lower())
    try:
        if key in cache_freq_brwac:
            return cache_freq_brwac[key]

        df_tmp = dff.loc[dff['Palavra'].str.lower() == w]
        df_tmp = df_tmp.loc[df_tmp['Tag'].str.lower() == tag.lower()]
        freq = df_tmp.iloc[0]['Frequência']
    except Exception as exc:
        print("freq-brwac", w, tag, "-->", exc)

    cache_freq_brwac[key] = freq
    return freq


# semantic similarity =================

def word2vec(text, embeddings):
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
    lsa_word_emb = word2vec(word, embeddings['lsa'])
    lsa_text_emb = word2vec(text, embeddings['lsa'])
    lsa_sim = similarity(lsa_word_emb, lsa_text_emb)

    ft_word_emb = word2vec(word, embeddings['fasttext'])
    ft_text_emb = word2vec(text, embeddings['fasttext'])
    ft_sim = similarity(ft_word_emb, ft_text_emb)

    w2v_word_emb = word2vec(word, embeddings['word2vec'])
    w2v_text_emb = word2vec(text, embeddings['word2vec'])
    w2v_sim = similarity(w2v_word_emb, w2v_text_emb)

    if lsa_sim > 1 or lsa_sim < -1:
        lsa_sim = 0

    if ft_sim > 1 or ft_sim < -1:
        ft_sim = 0

    if w2v_sim > 1 or w2v_sim < -1:
        w2v_sim = 0

    return round(lsa_sim, 5), round(ft_sim, 5), round(w2v_sim, 5)


embeddings = {}


def load_embeddings():
    lsa_embeddings_file = 'data/brwac_full_lsa_word_dict.pkl'

    with open(lsa_embeddings_file, 'rb') as f:
        embeddings['lsa'] = pickle.load(f)

    fasttext_embedding_file = 'data/bbp_fasttext_cbow_300d.txt'
    word2vec_embedding_file = 'data/bbp_word2vec_cbow_300d.txt'
    embeddings['fasttext'] = gensim.models.KeyedVectors.load_word2vec_format(fasttext_embedding_file)
    embeddings['word2vec'] = gensim.models.KeyedVectors.load_word2vec_format(word2vec_embedding_file)


# semantic similarity =====================


delaf = {'cache': {}}


def load_delaf():
    delaf['df'] = pd.read_csv('/home/sidleal/sid/usp/delaf/DELAF_PB_2018.dic', names=['word', 'info'])
    delaf['df']['sep'] = delaf['df']['info'].str.split(".")


def get_delaf_info(word):
    lemma, tag, morph = "", "", ""

    word = "%s" % word

    if word in delaf['cache']:
        word_cache = delaf['cache'][word]
        return word_cache[0], word_cache[1], word_cache[2]

    try:
        itens = delaf['df'].loc[delaf['df']['word'] == word]['sep']
        first = itens.iloc[0]
        lemma = first[0]
        tag_morph = first[1].split(":")
        tag = tag_morph[0]
        morph = ""

        if len(tag_morph) > 1:
            morph = tag_morph[1]

    except Exception as exc:
        print("get_delaf_info", word, "-->", exc)

    delaf['cache'][word] = [lemma, tag, morph]

    return lemma, tag, morph


def anonymize_participants():
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


def load_nlpnet_exceptions():
    ret = pd.read_csv('data/nlpnet_exceptions.tsv', sep='\t')
    return ret


nlpnet_exceptions_cache = {}


def get_nlpnet_exception(w):
    ret = 'ERR'

    if w in nlpnet_exceptions_cache:
        ret = nlpnet_exceptions_cache[w]
        return ret

    try:
        itens = nlpnet_exceptions.loc[nlpnet_exceptions['word'] == w]
        first = itens.iloc[0]
        ret = first['tag']

    except Exception as exc:
        # print("get_annotation", answer, "-->", exc)
        ret = 'ERR'

    nlpnet_exceptions_cache[w] = ret

    return ret


if __name__ == '__main__':
    print('--- Inicia processamento...')
    ini = time.time()

    print('--- Carrega o dataset principal...')
    df = load_main_dataset()
    print("Tempo: ", time.time() - ini)

    print('--- anonimização participantes...')
    participants = anonymize_participants()
    print("Tempo: ", time.time() - ini)

    print('--- Carregando embeddings...')
    load_embeddings()
    print("Tempo: ", time.time() - ini)

    print('--- Carregando delaf...')
    load_delaf()
    print("Tempo: ", time.time() - ini)

    print('--- Carregando answears annotation...')
    answers_annotation = load_answers_annotations()
    print("Tempo: ", time.time() - ini)

    print('--- Carregando exceções nlpnet...')
    nlpnet_exceptions = load_nlpnet_exceptions()
    print("Tempo: ", time.time() - ini)

    f = open("out/%s" % output_file, "w")
    write_output_header()

    text_id_refs, text_id_genres, paragraphs = get_text_ids_pars_and_genres()

    print('--- Executa PoS...')
    paragraphs_tags = get_pos_nlpnet_all_pars(paragraphs)
    print("Tempo: ", time.time() - ini)

    print('--- calcula tamanhos das sentenças...')
    sentence_lengths = get_sentence_lengths()
    print("Tempo: ", time.time() - ini)

    total_lines = 0
    wsidx, sent_length, word_place_in_sent = 0, 0, 0
    last_sent, last_text = 0, 0
    pos_words_ctl, cache_sim, cache_sim_ctx = {}, {}, {}
    preceding_passage, last_word_id = "", ""

    print('--- Inicia loop das palavras...')

    for i, w in enumerate(df['Palavra']):
        cloze_par_id = df['Parágrafo'][i]
        text_id = text_id_refs[cloze_par_id]
        text_par = paragraphs[cloze_par_id - 1]  # texto do parágrafo
        widx = df['Índice Palavra'][i]

        word_id = "UID_%s_%s" % (text_id, widx)

        part_name = df['Nome Participante'][i]
        part = participants[part_name]

        sent_idx = df['Sentença'][i]

        if sent_idx == last_sent and text_id == last_text:
            wsidx += 1
            word_place_in_sent = get_word_place_in_sent(wsidx, sent_length)
        else:
            wsidx = 1
            word_place_in_sent = 1
            sent_length = sentence_lengths['%s_%s' % (cloze_par_id, sent_idx)]

        if text_id != last_text:
            print(text_par)
            pos_words_ctl = {}
            first_word = text_par.split(' ')[0]
            preceding_passage = first_word

        last_sent = sent_idx
        last_text = text_id

        genre = text_id_genres[cloze_par_id]
        time_start = df['Tempo Início(ms)'][i]
        time_dig = df['Tempo Digitação(ms)'][i]
        total_time = time_start + time_dig

        w_raw = w
        a = df['Resposta'][i]
        w = clean_word(w)
        a = clean_word(a)

        if word_id == last_word_id: #  avoid repetitions
            continue
        last_word_id = word_id

        wtg = get_pos_nlpnet_word_ctl(pos_words_ctl, w)
        tag = "ERR"
        if wtg in paragraphs_tags[cloze_par_id - 1]:
            tag = paragraphs_tags[cloze_par_id - 1][wtg]

        if tag == "ERR":
            tag = get_nlpnet_exception(w)

        tag_desc = tag_map_nlpnet[tag]
        content_func = content_or_function_word(tag)

        print(">", text_id, sent_idx, widx, word_id, w, tag, a)

        # PoS
        w_lemma, w_tag_delaf, w_morph_delaf = get_delaf_info(w)

        a_lemma, a_tag_delaf, a_morph_delaf = '', '', ''
        a_correcao, a_tag, a_morph = get_annotation(a)
        if a_tag != '':
            a_lemma, a_tag_delaf, a_morph_delaf = '', a_tag, a_morph
        if a_correcao != '':
            a = a_correcao

        if a_tag_delaf == '':
            a_lemma, a_tag_delaf, a_morph_delaf = get_delaf_info(a)

        # contextual_fit / LSA_Context_Score
        preceding_passage = '%s %s' % (preceding_passage, w)
        if word_id in cache_sim_ctx:
            lsa_sim_ctx, ft_sim_ctx, w2v_sim_ctx \
                = cache_sim_ctx[word_id][0], cache_sim_ctx[word_id][1], cache_sim_ctx[word_id][2]
        else:
            lsa_sim_ctx, ft_sim_ctx, w2v_sim_ctx = get_similarity(w, preceding_passage)
            cache_sim_ctx[word_id] = [lsa_sim_ctx, ft_sim_ctx, w2v_sim_ctx]

        # semantic_relatedness / LSA_Response_Match_Score
        cache_sim_key = "%s_%s" % (w, a)
        if cache_sim_key in cache_sim:
            lsa_sim, ft_sim, w2v_sim \
                = cache_sim[cache_sim_key][0], cache_sim[cache_sim_key][1], cache_sim[cache_sim_key][2]
        else:
            lsa_sim, ft_sim, w2v_sim = get_similarity(w, a)
            cache_sim[cache_sim_key] = [lsa_sim, ft_sim, w2v_sim]

        freq_cb = get_freq_cb(df_freq_bra, w)
        freq_brwac = get_freq_brwac(df_freq_brwac, w, tag)


        ortho_match = 0
        if w == a:
            ortho_match = 1

        pos_match = 0
        if w_tag_delaf == a_tag_delaf:
            pos_match = 1

        morph_match = 0
        if w_morph_delaf == a_morph_delaf:
            morph_match = 1

        write_output_line(part, word_id, text_id, widx, sent_idx, wsidx, word_place_in_sent, sent_length, w_raw,
                          w, a, ortho_match, tag, content_func, tag_desc, w_tag_delaf, a_tag_delaf, pos_match,
                          w_morph_delaf, a_morph_delaf, morph_match, freq_cb, freq_brwac, genre, lsa_sim_ctx,
                          lsa_sim, ft_sim_ctx, ft_sim, w2v_sim_ctx, w2v_sim, time_start, time_dig, total_time)

        total_lines += 1

    print("Total lines:", total_lines)

    print("Tempo Exec: ", (time.time() - ini)/60)

    f.close()
