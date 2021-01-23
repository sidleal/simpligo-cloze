import math
import pandas as pd
import itertools


df_pars_eye = pd.read_csv('data/50pars_eye.txt', sep='\t', escapechar='\\')

df = pd.read_csv('out/cloze_predict_38_FULL.tsv', sep='\t', quoting=3)
df = df.fillna('')
df = df.sort_values(by=['Word_Unique_ID'])

total_words = 0
last_word_id = ''
word_list = []
total_resp_count, orto_match_count, word_len = 0, 0, 0
pos_match_count, inflection_match_count = 0, 0
text_id, genre, text, word, word_clean = '', '', '', '', ''
word_num, sentence_num, word_in_sent, word_place = '', '', '', ''
word_pos_tag, word_cont_or_func, word_pos_desc = '', '', ''
resps, semantic_context_score, freq, times = {}, {}, {}, {}

def fill_word_item():
    ordered_answers = {key: val for key, val in
                       sorted(resps.items(), key=lambda item: item[1], reverse=True)}

    ret = []
    for resp, resp_count in ordered_answers.items():
        if resp == '':
            continue
        resp_proportion = round(resp_count / total_resp_count, 3)
        word_item = [last_word_id, text_id, text,
                     word_num, sentence_num, word_in_sent,
                     word_clean, resp,
                     resp_count, total_resp_count, resp_proportion]

        ret.append(word_item)
        print(word_item)

    return ret


for i, word_id in enumerate(df['Word_Unique_ID']):

    if last_word_id != word_id:
        # mudou palavra, consolida e joga pra lista
        if last_word_id != '':
            word_list = word_list + fill_word_item()

        #zera tudo.. mudou de palavra
        total_resp_count, orto_match_count = 0, 0
        pos_match_count, inflection_match_count = 0, 0
        resps, semantic_context_score, freq, times = {}, {}, {}, {}

    word = df['Word'].iloc[i]
    word_clean = df['Word_Cleaned'].iloc[i]
    text_id = df['Text_ID'].iloc[i]
    text = df_pars_eye['Paragraph'][text_id-1]
    word_num = df['Word_Number'].iloc[i]
    sentence_num = df['Sentence_Number'].iloc[i]
    word_in_sent = df['Word_In_Sentence_Number'].iloc[i]

    answer = df['Response_Cleaned'].iloc[i]
    if answer in resps:
        resps[answer] += 1
    else:
        resps[answer] = 1

    total_resp_count += 1
    last_word_id = word_id

word_list = word_list + fill_word_item()


f = open("out/cloze_predict38_norms.tsv", "w")

header = 'Word_Unique_ID\tText_ID\tText\t' \
         'Word_Number\tSentence_Number\tWord_In_Sentence_Number\t' \
         'Word\tResponse\t' \
         'Response_Count\tTotal_Response_Count\tResponse_Proportion\n'

print(header)
f.write(header)

for word_info in word_list:
    line = ""
    for col in word_info:
        line = '%s%s\t' % (line, col)


    line = line[:-1] + '\n'
    f.write(line)
    total_words += 1
    print(line)

f.close()

print("total words:", total_words)
