import math

import pandas as pd

df_freq_brwac = pd.read_csv('data/lista_brWaC_geral_v3_nlpnet.tsv', sep='\t')
df_freq_brasileiro = pd.read_csv('data/wl_cb_full_1gram_sketchengine.txt', sep='\t',
                                 names=["Word", "Frequency"], dtype={"Word": "string", "Frequency": int},
                                 keep_default_na=False, na_values=['_'])

if __name__ == '__main__':
    brwac_size = df_freq_brwac['Frequência'].sum()

    brwac_file = open("out/brWaC_Normalized_Frequencies.tsv", "w")
    brwac_file.write("Word\tTag\tFrequency_FPM\tFrequency_Log\n")

    for i, w in enumerate(df_freq_brwac['Palavra']):
        print('%s - %s - %s' % (w, df_freq_brwac['Tag'][i], df_freq_brwac['Frequência'][i]))
        freq = df_freq_brwac['Frequência'][i]
        freq_fpm = freq * 1000000 / brwac_size
        freq_log = math.log(freq_fpm, 10)+3
        brwac_file.write("%s\t%s\t%.05f\t%.05f\n" % (w, df_freq_brwac['Tag'][i], freq_fpm, freq_log))

    brwac_file.close()

    brasileiro_size = df_freq_brasileiro['Frequency'].sum()
    brasileiro_file = open("out/brasileiro_Normalized_Frequencies.tsv", "w")
    brasileiro_file.write("Word\tFrequency_FPM\tFrequency_Log\n")

    brasileiro_map = {}
    for i, w in enumerate(df_freq_brasileiro['Word']):
        if w.lower() in brasileiro_map:
            brasileiro_map[w.lower()] += df_freq_brasileiro['Frequency'][i]
        else:
            brasileiro_map[w.lower()] = df_freq_brasileiro['Frequency'][i]

    for w, freq in brasileiro_map.items():
        print('%s - %s' % (w, freq))
        freq_fpm = freq * 1000000 / brasileiro_size
        freq_log = math.log(freq_fpm, 10)+3
        brasileiro_file.write("%s\t%.05f\t%.05f\n" % (w, freq_fpm, freq_log))

    brasileiro_file.close()