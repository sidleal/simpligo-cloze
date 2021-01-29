import time
import numpy as np
import string

from scipy import spatial
import gensim


def getVectorsEmbeddings(text, embeddings):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    tokens = text.split()

    dim = embeddings['word'].size
    word_vec = []
    for word in tokens:
        if word.lower() in embeddings:
            word_vec.append(embeddings[word.lower()])
        else:
            word_vec.append(np.random.uniform(-0.001, 0.001, dim))

    if len(word_vec) == 0:
        word_vec.append(np.random.uniform(-0.001, 0.001, dim))

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
        print("---", word, text, ft_word_emb, ft_text_emb)
        print(exc)
        exit(1)

    if ft_sim > 1:
        ft_sim = 1
    if ft_sim < 0:
        ft_sim = 0

    context = '[CLS] %s [MASK] [SEP]' % text
    bert_score = calc_score_task1(context, word, tokenizer, model, debug=True)
    ret = bert_score
    if ret is None:
        print("bert - none: ", word, ft_sim)
        ret = ft_sim

    return ret


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
    ret = bert_score
    if ret is None:
        print("bert - none: ", word, answer, ft_sim)
        ret = ft_sim

    return ret


embeddings = {}


def load_embeddings():
    # Downloaded from:
    # https://www.inf.pucrs.br/linatural/wordpress/pucrs-bbp-embeddings/
    # or
    # http://www.nilc.icmc.usp.br/embeddings
    fasttext_embedding_file = 'data/bbp_fasttext_cbow_300d.txt'
    embeddings['fasttext'] = gensim.models.KeyedVectors.load_word2vec_format(fasttext_embedding_file)


#BEGIN BERT
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# if you want to have more information on what's happening, activate the logger as follows
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

    predictions_candidates = predictions[0][0][masked_index]

    predicted_index = torch.argmax(predictions_candidates).item()

    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

    # if word dont exist return None
    if expected_token == '[UNK]':
        return None

    predictions_candidates = predictions_candidates.cpu().numpy()
    target_bert_confiance = predictions_candidates[target_index]
    predicted_bert_confience = predictions_candidates[predicted_index]

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

  # Predict all tokens
  with torch.no_grad():
      predictions = model(tokens_tensor, segments_tensors)

  predictions_candidates = predictions[0][0][masked_index].cpu().numpy()

  # get words again
  expected_token = tokenizer.convert_ids_to_tokens([target_index])[0]
  predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

  # if word dont exist return None
  if predicted_token == '[UNK]' or expected_token == '[UNK]':
    return None

  target_bert_confiance = predictions_candidates[target_index]
  predicted_bert_confience = predictions_candidates[predicted_index]

  score = target_bert_confiance - predicted_bert_confience if target_bert_confiance > predicted_bert_confience else predicted_bert_confience - target_bert_confiance

  if debug:
    print("predicted token ---> ", predicted_token, predicted_bert_confience)
    print("expected token  ---> ",expected_token , target_bert_confiance)
    print("Score:", score)

  return score

#END BERT


if __name__ == '__main__':

    print('--- Begin process...')
    ini = time.time()

    print('--- Loading embeddings...')
    load_embeddings()
    print("Time: ", time.time() - ini)

    print('--- Loading BERT...')
    tokenizer, model = load_bert_model()
    print("Time: ", time.time() - ini)

    # preceding_passage = "Especificamente, a capacidade impressionante desses insetos de se espremerem por qualquer espaço e aguentarem"
    # target_word = "pressões"
    # response_word = "altas"
    preceding_passage = "Pesquisadores americanos passaram os últimos tempos estudando um assunto bastante"
    target_word = "peculiar"
    response_word = "importante"

    # contextual_fit / LSA_Context_Score
    sim_ctx = get_similarity(target_word, preceding_passage)
    sim_resp_ctx = get_similarity(response_word, preceding_passage)

    # semantic_relatedness / LSA_Response_Match_Score
    sim_resp_match = get_similarity_match(target_word, response_word, preceding_passage)

    print("Contextual Fit Target Word: ", sim_ctx)
    print("Contextual Fit Response: ", sim_resp_ctx)
    print("Semantic Relatedness Response vs Target: ", sim_resp_match)

    #Expected output:
    #Contextual Fit Target Word:  9.746683
    #Contextual Fit Response:  9.01776
    #Semantic Relatedness Response vs Target:  0.7289231