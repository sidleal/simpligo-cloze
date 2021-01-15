import pandas as pd

output_file = "cloze_predict_38_FULL.tsv"


#sub_df_age['idade'].mean() + 2.5 * sub_df_age['idade'].std()
#43.27398918303636
parts_more_than_43 = [338, 70, 77, 315, 178, 245, 118, 371, 93, 279, 221, 34, 28]
parts_no_attention = [29, 34, 73, 304, 321, 390, 411, 243, 396, 391]

invalid_resps_pars = [
    [381, 39], [381, 15], [381, 29], [378, 43], [84, 45], [112, 48], [179, 31], [84, 20], [177, 27], [179, 42],
    [84, 31], [381, 30], [355, 44], [84, 36], [177, 24], [177, 2], [177, 25], [84, 33], [154, 2], [154, 28],
    [407, 22], [355, 37], [154, 48], [141, 21], [55, 38], [91, 22], [55, 4], [3, 13], [299, 5], [177, 41],
    [91, 49], [299, 44], [381, 43], [55, 11], [154, 16], [56, 44], [31, 32], [296, 43], [141, 46], [197, 8],
    [63, 22], [4, 10], [270, 7], [384, 23], [389, 27], [400, 31]
]

if __name__ == '__main__':

    out1 = pd.read_csv("out/%s_%s" % ('t1', output_file), sep='\t', quoting=3)
    out2 = pd.read_csv("out/%s_%s" % ('t2', output_file), sep='\t', quoting=3)
    out3 = pd.read_csv("out/%s_%s" % ('t3', output_file), sep='\t', quoting=3)
    out4 = pd.read_csv("out/%s_%s" % ('t4', output_file), sep='\t', quoting=3)
    out5 = pd.read_csv("out/%s_%s" % ('t5', output_file), sep='\t', quoting=3)

    out = out1.append(out2, ignore_index=True)
    out = out.append(out3, ignore_index=True)
    out = out.append(out4, ignore_index=True)
    out = out.append(out5, ignore_index=True)

    for part in parts_more_than_43:
        out = out.drop(out.query("Participant == %s" % part).index)

    for part in parts_no_attention:
        out = out.drop(out.query("Participant == %s" % part).index)

    for part_par in invalid_resps_pars:
        out = out.drop(out.query("Participant == %s and Text_ID == %s" % (part_par[0], part_par[1])).index)

    #problema alinhamento
    out = out.drop(out.query("Participant == 372 and Text_ID == 26 and Word_Number > 41").index) # 3 palavras
    out = out.drop(out.query("Participant == 328 and Text_ID == 33 and Word_Number > 51").index) # 10 palavras

    max_score = out['Semantic_Word_Context_Score'].max()
    if out['Semantic_Response_Match_Score'].max() > max_score:
        max_score = out['Semantic_Response_Match_Score'].max()
    if out['Semantic_Response_Context_Score'].max() > max_score:
        max_score = out['Semantic_Response_Context_Score'].max()

    out['Semantic_Word_Context_Score'] = round(1 - (
            out['Semantic_Word_Context_Score'] / max_score), 3)
    out['Semantic_Response_Match_Score'] = round(1 - (
            out['Semantic_Response_Match_Score'] / max_score), 3)
    out['Semantic_Response_Context_Score'] = round(1 - (
            out['Semantic_Response_Context_Score'] / max_score), 3)

    out.loc[out['Semantic_Word_Context_Score'] > 1, "Semantic_Word_Context_Score"] \
        = round(out['Semantic_Word_Context_Score_ft'], 3)
    out.loc[out['Semantic_Response_Match_Score'] > 1, "Semantic_Response_Match_Score"] \
        = round(out['Semantic_Response_Match_Score_ft'], 3)
    out.loc[out['Semantic_Response_Context_Score'] > 1, "Semantic_Response_Context_Score"] \
        = round(out['Semantic_Response_Context_Score_ft'], 3)

    out = out.drop('Semantic_Word_Context_Score_ft', axis='columns')
    out = out.drop('Semantic_Response_Match_Score_ft', axis='columns')
    out = out.drop('Semantic_Response_Context_Score_ft', axis='columns')

    df_text_sizes = out.groupby('Text_ID')['Word_Unique_ID'].nunique().to_frame()
    df_text_sizes.columns = ['text_size']
    df1 = pd.merge(out, df_text_sizes, on='Text_ID')
    df_answers_sizes = df1.groupby(['Participant', 'Text_ID', 'text_size']).size().to_frame()
    df_answers_sizes.columns = ['answers_size']
    df2 = pd.merge(df1, df_answers_sizes, on=['Participant', 'Text_ID'])
    df2['Resp_rate'] = round(df2['answers_size'] * 100 / df2['text_size'], 2)

    out = df2.drop('text_size', axis='columns')
    out = out.drop('answers_size', axis='columns')

    out = out.sort_values(['Text_ID', 'Participant', 'Word_Number'])
    out.to_csv("out/%s" % output_file, sep='\t', quoting=3, index=False)
