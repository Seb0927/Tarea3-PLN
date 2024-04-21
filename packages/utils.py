from tqdm.auto import tqdm
from time import sleep
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import torch

def build_matrix_embeddings(path, num_tokens, embedding_dim, word_index):
    """
        Função para carregar arquivos pre-treinados em memória
    """

    hits, misses = 0, 0
    embeddings_index = {}

    print('Cargando archivo...')

    sleep(0.5)

    for line in tqdm(open(path, encoding='utf-8')):
        word, coefs = line.split(maxsplit=1)
        embeddings_index[word] = np.fromstring(coefs, "f", sep=" ")

    print("Encontrado %s Word Vectors." % len(embeddings_index))

    sleep(0.5)

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))

    for word, i in tqdm(word_index.items()):
        if i >= num_tokens:
            continue
        try:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                embedding_vector = embeddings_index.get(str(word).lower())
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                    hits += 1
                else:
                    embedding_vector = embeddings_index.get(str(word).upper())
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
                        hits += 1
                misses += 1
        except:
            embedding_matrix[i] = embeddings_index.get('UNK')

    print("Convertidos: %d Tokens | Perdidos: %d Tokens" % (hits, misses))

    return embedding_matrix

def plot_model_performance(train_loss, train_acc, train_val_loss, train_val_acc):
    """ Plot model loss and accuracy through epochs. """
    blue= '#34495E'
    green = '#2ECC71'
    orange = '#E23B13'
    
    # plot model loss
    #fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_loss) + 1), train_loss, blue, linewidth=5, label='training')
    plt.plot(range(1, len(train_val_loss) + 1), train_val_loss, green, linewidth=5, label='validation')
    plt.xlabel('# epoch')
    plt.ylabel('loss')
    plt.tick_params('y')
    plt.legend(loc='upper right', shadow=False)
    plt.title('Model loss through #epochs', color=orange, fontweight='bold')
    
    # plot model accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_acc) + 1), train_acc, blue, linewidth=5, label='training')
    plt.plot(range(1, len(train_val_acc) + 1), train_val_acc, green, linewidth=5, label='validation')
    plt.xlabel('# epoch')
    plt.ylabel('accuracy')
    plt.tick_params('y')
    plt.legend(loc='lower right', shadow=False)
    plt.title('Model accuracy through #epochs', color=orange, fontweight='bold')
    
    #fig.savefig('Plot/training/training-mb-00.png', bbox_inches='tight')

def logits_to_tokens(sequences, indexa):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(indexa[categorical])
 
        token_sequences.append(token_sequence)
 
    return token_sequences

def report_to_df(report):
    report = [x.split(' ') for x in report.split('\n')]
    header = ['Class Name']+[x for x in report[0] if x!='']
    values = []
    for row in report[1:-5]:
        row = [value for value in row if value!='']
        if row!=[]:
            values.append(row)
    df = pd.DataFrame(data = values, columns = header)
    return df

############### PYTORCH utils #################

def build_vocab(sentences, word2idx):
    # Create word-to-index mapping
    for sentence, tags in sentences:  
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    return word2idx

def getData(vector):
    data_vector = []
    for sentence in vector:
        sentences = []
        labels    = []
        for word, tag in sentence:
            sentences.append(word)
            labels.append(tag)
        data_vector.append((sentences, labels))
    
    return data_vector

def convert2inputs(seqs, to_ids, maxlen, device):
    #pad_id = 0
    #unk_id = 1
    ids = []
    ids = [to_ids[w] for w in seqs]
    #for seq in seqs:
    #    if len(seq)>=maxlen: # truncation
    #        ids.append([to_ids[w] if w in to_ids else unk_id for w in seq[:maxlen]])
    #    else: # padding
    #        ids.append([to_ids[w] if w in to_ids else unk_id for w in seq] + [0]*(maxlen-len(seq)))

    return torch.tensor(ids, device=device, dtype=torch.long)

def createDataloader(vector, word2idx, tag2idx, maxlen, batch_size, device, shuffle=False):
    sentences = []
    labels    = []
    for sentencia, tags in vector:
        sentences += sentencia
        labels    += tags
        
    inps = convert2inputs(sentences, word2idx, maxlen, device)
    trgs = convert2inputs(labels, tag2idx, maxlen, device)

    inp_dset    = torch.utils.data.TensorDataset(inps, trgs)
    inp_dloader = torch.utils.data.DataLoader(inp_dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=0)
    return inp_dloader

# Draw training curve
def plot_metric(df_history, metric):
    plt.figure()

    train_metrics = df_history[metric]
    val_metrics = df_history['val_' + metric]  #

    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')  #

    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])

    plt.savefig(imgs_dir + metric + '.png')  # Save Pictures
    plt.show()

 # Time stamp
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 1 + " %s"%nowtime)