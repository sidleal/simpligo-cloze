import time


def calc_score_tarefa1(text, target_word, tokenizer, model, debug=False):
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


def calc_score(text, target_word, predicted_word, tokenizer, model, debug=False):

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



import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased')

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('neuralmind/bert-large-portuguese-cased')
model.eval()


# text = '[CLS] Especificamente, a capacidade impressionante desses insetos de se espremerem por qualquer espaço e aguentarem [MASK] de até 900 vezes seu próprio peso sem sofrer grandes danos [SEP]'
text = '[CLS] Especificamente, a capacidade impressionante desses insetos de se espremerem por qualquer espaço e aguentarem [MASK] temperaturas [SEP]'

target_word = 'pressões'
predicted_word = 'altas'

# calculate score
score = calc_score_tarefa1(text, target_word, tokenizer, model, debug=True)
print("Score: ", target_word, score)


# calculate score
score = calc_score(text, target_word, predicted_word, tokenizer, model, debug=True)
print("Score: ", target_word, predicted_word, score)

exit(1)

#///////////////////
import pandas as pd

output_file = "bert_out.tsv"
process_date = "2020_10_31"

def load_main_dataset():
    df_all = pd.read_csv('../data/cloze_all_%s.csv' % process_date)
    print("total:", df_all['Palavra'].count())

    return df_all


def write_output_header():
    header = 'Word_Unique_ID\tContext\tTarget\tResponse\tScore T1\tScore T2\tScore T3\n'
    print(header)
    f.write(header)


def write_output_line(wuid, preceding_passage, word, resp, score_t1, score_t2, score_t3):
    line = '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
           % (wuid, preceding_passage, word, resp, score_t1, score_t2, score_t3)
    f.write(line)



def get_text_ids_pars_and_genres():
    df_pars_eye = pd.read_csv('../data/50pars_eye.txt', sep='\t', escapechar='\\')
    df_pars_cloze = pd.read_csv('../data/50pars_cloze.txt', sep='\t', escapechar='\\')

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



def get_sentence_lengths():
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


def load_answers_annotations():
    answers_annotation = pd.read_csv('../data/answers_annotation.tsv', sep='\t')
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


print('--- Inicia processamento...')
ini = time.time()

print('--- Carrega o dataset principal...')
df = load_main_dataset()
print("Tempo: ", time.time() - ini)

print('--- Carregando answears annotation...')
answers_annotation = load_answers_annotations()
print("Tempo: ", time.time() - ini)

f = open("../out/%s" % output_file, "w")
write_output_header()

text_id_refs, text_id_genres, paragraphs = get_text_ids_pars_and_genres()

print('--- calcula tamanhos das sentenças...')
sentence_lengths, sentences = get_sentence_lengths()
print("Tempo: ", time.time() - ini)

total_lines_count = 1
wsidx, sent_length, word_place_in_sent = 0, 0, 0
last_sent, last_text = 0, 0
pos_words_ctl, cache_sim, cache_sim_ctx = {}, {}, {}
preceding_passage, last_word_id, last_word = "", "", ""
wtg = 0
sentence, tags_pos_palavras, cache_tags_pos_palavras_sents = {}, {}, {}
text_sent, first_word = "", ""
first_word_of_a_text = False

cache_scores = {}

print(sentences)

print('--- Inicia loop das palavras...')

total_lines = df['Palavra'].count()

for i, w in enumerate(df['Palavra']):
    cloze_par_id = df['Parágrafo'][i]
    text_id = text_id_refs[cloze_par_id]
    text_par = paragraphs[cloze_par_id - 1]  # texto do parágrafo
    widx = df['Índice Palavra'][i]

    word_id = "UID_%s_%s" % (text_id, widx)

    sent_idx = df['Sentença'][i]
    sent_str_id = '%s_%s' % (cloze_par_id, sent_idx)

    if word_id != last_word_id:
        first_word_of_a_text = False
        preceding_passage = '%s %s' % (preceding_passage, last_word)
        if sent_idx == last_sent and text_id == last_text:
            wsidx += 1
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
            text_sent += "%s " % first_word
        print(sentence)

        for k, v in sentence.items():
            text_sent += "%s " % v

        print("-------", text_sent)

    genre = text_id_genres[cloze_par_id]
    time_start = df['Tempo Início(ms)'][i]
    time_dig = df['Tempo Digitação(ms)'][i]
    total_time = time_start + time_dig

    w_raw = w
    a = df['Resposta'][i]
    a_raw = a
    w = clean_word(w)
    a = clean_word(a)

    print(round((time.time() - ini)/60, 2), total_lines_count, "/", total_lines,
          ">", text_id, sent_idx, widx, word_id, w, a)

    last_sent = sent_idx
    last_text = text_id
    last_word_id = word_id
    last_word = w

    print(preceding_passage, w_raw)

    a_to_sim = a_raw
    a_lemma, a_tag, a_morph = '', '', ''
    a_correcao, a_tag, a_morph = get_annotation(a)
    if a_correcao != '':
        a = a_correcao
        a_to_sim = a_correcao
    if a_tag == 'RANDOM':
        continue

    context = '[CLS] %s [MASK] [SEP]' % preceding_passage
    print(context)

    score_t1 = 0
    if "%s_%s" % (word_id, w) in cache_scores:
        score_t1 = cache_scores["%s_%s" % (word_id, w)]
    else:
        score_t1 = calc_score_tarefa1(context, w_raw, tokenizer, model, debug=False)
        cache_scores["%s_%s" % (word_id, w)] = score_t1

    score_t2 = 0
    if "%s_%s_%s" % (word_id, w, a) in cache_scores:
        score_t2 = cache_scores["%s_%s_%s" % (word_id, w, a)]
    else:
        score_t2 = calc_score(context, w_raw, a_to_sim, tokenizer, model, debug=True)
        cache_scores["%s_%s_%s" % (word_id, w, a)] = score_t2

    score_t3 = 0
    if "%s_%s" % (word_id, a) in cache_scores:
        score_t3 = cache_scores["%s_%s" % (word_id, a)]
    else:
        score_t3 = calc_score_tarefa1(context, a_to_sim, tokenizer, model, debug=True)
        cache_scores["%s_%s" % (word_id, a)] = score_t3

    print("Score: ", w_raw, score_t1, a_raw, score_t2, score_t3)

    # a_tag, a_lemma, a_morph = get_pos_palavras_answer_cache(preceding_passage, word_id, a_raw)

    write_output_line(word_id, preceding_passage, w_raw, a_raw, score_t1, score_t2, score_t3)

    total_lines_count += 1

print("Total lines:", total_lines_count)

print("Tempo Exec: ", (time.time() - ini)/60)

f.close()


