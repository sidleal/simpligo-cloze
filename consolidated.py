import math
import pandas as pd
import itertools


df_pars_eye = pd.read_csv('data/50pars_eye.txt', sep='\t', escapechar='\\')

df = pd.read_csv('out/cloze_predict_39_FULL.tsv', sep='\t', quoting=3)
df = df.fillna('')
df = df.sort_values(by=['Word_Unique_ID'])

total_words = 0
last_word_id, last_text_id = '', ''
word_list = []
resp_count, orto_match_count, word_len = 0, 0, 0
pos_match_count, inflection_match_count = 0, 0
text_id, genre, text, word, word_clean = '', '', '', '', ''
word_num, sentence_num, word_in_sent, word_place = '', '', '', ''
word_pos_tag, word_cont_or_func, word_pos_desc = '', '', ''
resps, semantic_context_score, freq, times = {}, {}, {}, {}
last_entropy = 0

def fill_word_item():
    global last_entropy, resp_count

    ordered_answers = {key: val for key, val in
                       sorted(resps.items(), key=lambda item: item[1], reverse=True)}
    top_10_answers = dict(itertools.islice(ordered_answers.items(), 10))

    unique_count = len(ordered_answers)

    orto_match = round(orto_match_count / resp_count, 3)
    modal_response = list(ordered_answers)[0]
    is_modal_resp = 0
    if word_clean == modal_response:
        is_modal_resp = 1
    modal_resp_count = resps[modal_response]

    certainty = round(modal_resp_count / resp_count, 3)

    pos_match = round(pos_match_count / resp_count, 3)
    inflection_match = round(inflection_match_count / resp_count, 3)

    sem_word_ctx = semantic_context_score['Semantic_Word_Context_Score']
    sem_resp_match = round(semantic_context_score['Semantic_Response_Match_Score'] / resp_count, 3)
    sem_resp_ctx = round(semantic_context_score['Semantic_Response_Context_Score'] / resp_count, 3)

    freq_brasil_fpm = freq['Freq_brasileiro_fpm']
    freq_brwac_fpm = freq['Freq_brWaC_fpm']
    freq_brasil_log = freq['Freq_brasileiro_log']
    freq_brwac_log = freq['Freq_brWaC_log']

    time_to_start = int(times['Time_to_Start'] / resp_count)
    type_time = int(times['Typing_Time'] / resp_count)
    total_time = int(times['Total_time'] / resp_count)

    #surprisal
    orto_match_above_zero = orto_match
    if orto_match_above_zero  <= 0:
        orto_match_above_zero = 0.0115 # metade do menor valor: 0.23
    surprisal = math.log(orto_match_above_zero, 10) * -1
    surprisal = round(surprisal, 3)

    #entropy reduction
    cur_entropy = 0
    for k, resp_qty in resps.items():
        cur_entropy += (resp_qty/resp_count) * math.log2(resp_qty/resp_count) * -1

    entropy_reduction = cur_entropy - last_entropy
    # if entropy_reduction > 0:
    #     entropy_reduction = 0
    entropy_reduction = round(entropy_reduction, 3)

    last_entropy = cur_entropy

    if word_num == 1: #first word
        sem_word_ctx, sem_resp_match, sem_resp_ctx, certainty = 0, 0, 0, 0
        modal_resp_count, unique_count, resp_count, inflection_match = 0, 0, 0, 0
        top_10_answers = {}

    word_item = [last_word_id, text_id, text, genre,
                 word_num, sentence_num, word_in_sent, word_place,
                 word, word_clean, word_len, resp_count, unique_count, orto_match,
                 is_modal_resp, modal_response, modal_resp_count, certainty,
                 word_pos_tag, word_cont_or_func, word_pos_desc, pos_match, word_inflection, inflection_match,
                 sem_word_ctx, sem_resp_match, sem_resp_ctx,
                 freq_brwac_fpm, freq_brasil_fpm, freq_brwac_log, freq_brasil_log,
                 surprisal, entropy_reduction,
                 time_to_start, type_time, total_time,
                 top_10_answers]

    print(word_item)

    return word_item


for i, word_id in enumerate(df['Word_Unique_ID']):

    text_id = df['Text_ID'].iloc[i]

    if last_word_id != word_id:
        # mudou palavra, consolida e joga pra lista
        if last_word_id != '':
            word_list.append(fill_word_item())

        #zera tudo.. mudou de palavra
        resp_count, orto_match_count = 0, 0
        pos_match_count, inflection_match_count = 0, 0
        resps, semantic_context_score, freq, times = {}, {}, {}, {}
        semantic_context_score['Semantic_Word_Context_Score'] = 0
        semantic_context_score['Semantic_Response_Match_Score'] = 0
        semantic_context_score['Semantic_Response_Context_Score'] = 0
        times['Time_to_Start'], times['Typing_Time'], times['Total_time'] = 0, 0, 0
        delaf_tags, delaf_morphs = {}, {}

        if last_text_id != text_id:
            last_entropy = 0

    word = df['Word'].iloc[i]
    word_clean = df['Word_Cleaned'].iloc[i]
    word_len = df['Word_Length'].iloc[i]
    text_id = df['Text_ID'].iloc[i]
    genre = df['Genre'].iloc[i]
    text = df_pars_eye['Paragraph'][text_id-1]
    word_num = df['Word_Number'].iloc[i]
    sentence_num = df['Sentence_Number'].iloc[i]
    word_in_sent = df['Word_In_Sentence_Number'].iloc[i]
    word_place = df['Word_Place_In_Sent'].iloc[i]
    sent_length = df['Sent_Length'].iloc[i]

    word_pos_tag = df['POS'].iloc[i]
    word_cont_or_func = df['Word_Content_Or_Function'].iloc[i]
    word_pos_desc = df['Word_POS'].iloc[i]
    word_inflection = df['Word_Inflection'].iloc[i]

    pos_match_count += df['POSMatch'].iloc[i]
    inflection_match_count += df['InflectionMatch'].iloc[i]

    semantic_context_score['Semantic_Word_Context_Score'] = df['Semantic_Word_Context_Score'].iloc[i]
    semantic_context_score['Semantic_Response_Match_Score'] += df['Semantic_Response_Match_Score'].iloc[i]
    semantic_context_score['Semantic_Response_Context_Score'] += df['Semantic_Response_Context_Score'].iloc[i]

    freq['Freq_brasileiro_fpm'] = df['Freq_brasileiro_fpm'].iloc[i]
    freq['Freq_brWaC_fpm'] = df['Freq_brWaC_fpm'].iloc[i]
    freq['Freq_brasileiro_log'] = df['Freq_brasileiro_log'].iloc[i]
    freq['Freq_brWaC_log'] = df['Freq_brWaC_log'].iloc[i]

    times['Time_to_Start'] += df['Time_to_Start'].iloc[i]
    times['Typing_Time'] += df['Typing_Time'].iloc[i]
    times['Total_time'] += df['Total_time'].iloc[i]

    answer = df['Response_Cleaned'].iloc[i]
    if answer in resps:
        resps[answer] += 1
    else:
        resps[answer] = 1

    if word_clean == answer:
        orto_match_count += 1

    resp_count += 1
    last_word_id = word_id
    last_text_id = text_id

word_list.append(fill_word_item())


f = open("out/cloze_predict40_consolidada.tsv", "w")

header = 'Word_Unique_ID\tText_ID\tText\tGenre\t' \
         'Word_Number\tSentence_Number\tWord_In_Sentence_Number\tWord_Place_In_Sent\t' \
         'Word\tWord_Cleaned\tWord_Length\tTotal_Response_Count\tUnique_Count\tOrthographicMatch\t' \
         'IsModalResponse\tModalResponse\tModalResponseCount\tCertainty\t' \
         'PoS\tWord_Content_Or_Function\tWord_PoS\tPOSMatch\tWord_Inflection\tInflectionMatch\t' \
         'Semantic_Word_Context_Score\tSemantic_Response_Match_Score\tSemantic_Response_Context_Score\t' \
         'Freq_brWaC_fpm\tFreq_Brasileiro_fpm\tFreq_brWaC_log\tFreq_Brasileiro_log\t' \
         'Surprisal\tEntropy_Reduction\t' \
         'Time_to_Start\tTyping_Time\tTotal_time\tTop_10_resp\n'
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
