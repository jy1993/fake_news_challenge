import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
from nltk import tokenize
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


def load_word2vec(filename, stopwords):
	word2vec = {}
	vecs = []
	vecs.append([0] * 100)
	with open(filename, encoding='utf-8') as f:
		for line in f.readlines():
			terms = line.strip().split()
			if len(terms) != 101:
				#print(terms)
				continue
			w, vec = terms[0], terms[1:]
			if w in stopwords:
				continue
			word2vec[w] = [float(v) for v in vec]
			vecs.append([float(v) for v in vec])

	return word2vec, np.array(vecs)


def get_head_and_tail(alist, num=100):
	'''
	get the first and last num eles from a list
	'''
	if len(alist) < 2 * num:
		return alist
	else:
		head = alist[:num]
		tail = alist[-num:]

	return head + tail


def get_score(y_true, y_pred):
	score = 0
	#label_2_id = {'agree': 0, 'disagree': 1, 'discuss':2, 'unrelated': 3}
	for (t, p) in zip(y_true, y_pred):
		if t == p:
			score += 0.25
			if t != 3:
				score += 0.5

		if t != 3 and p != 3:
			score += 0.25

	return score

def get_batch(filename, batch_size=64, max_len_h=50, max_len_b=100, data_aug=False):
	# load headline, body, stance from a file
	data = pickle.load(open(filename, 'rb'))
	headlines = np.array(data['h'])
	bodies = np.array(data['b'])
	y = np.array(data['y'])

	# sorting according to the articleBodies length
	len_b = [len(v) for v in bodies]
	# print('articleBodies length distriubtion:')
	# print(np.percentile(len_b, [0, 50, 95, 99, 100]))

	len_h = [len(v) for v in headlines]
	# print('headlines length distriubtion:')
	# print(np.percentile(len_h, [0, 50, 95, 99, 100]))
	# print('*' * 100)

	indices = np.argsort(len_b)

	sorted_h = headlines[indices]
	sorted_b = bodies[indices]
	sorted_y = y[indices]

	assert len(sorted_h) == len(sorted_b) == len(sorted_y)
	
	# using the first 200 words in articleBodies
	for i in range(0, len(sorted_h), batch_size):
		batch_h = sorted_h[i:i+batch_size]
		batch_b = sorted_b[i:i+batch_size]
		batch_y = sorted_y[i:i+batch_size]

		# get max length of headlines and bodies in the batch
		len_batch_h = [len(v) for v in batch_h]
		len_batch_b = [len(v) for v in batch_b]
		maxlen1 = max_len_h if max_len_h < max(len_batch_h) else max(len_batch_h)
		maxlen2 = max_len_b if max_len_b < max(len_batch_b) else max(len_batch_b)

		# padding to the max length
		batch_h = pad_sequences(batch_h, maxlen1, padding='pre', truncating='post')
		batch_b = pad_sequences(batch_b, maxlen2, padding='pre', truncating='post')
		yield (batch_h, batch_b, batch_y)
	
	# using the last 200 words in articleBodies
	if data_aug:
		for i in range(0, len(sorted_h), batch_size):
			batch_h = sorted_h[i:i+batch_size]
			batch_b = sorted_b[i:i+batch_size]
			batch_y = sorted_y[i:i+batch_size]

			# get max length of headlines and bodies in the batch
			len_batch_h = [len(v) for v in batch_h]
			len_batch_b = [len(v) for v in batch_b]
			maxlen1 = max_len_h if max_len_h < max(len_batch_h) else max(len_batch_h)
			maxlen2 = max_len_b if max_len_b < max(len_batch_b) else max(len_batch_b)

			# padding to the max length
			batch_h = pad_sequences(batch_h, maxlen1, padding='pre', truncating='pre')
			batch_b = pad_sequences(batch_b, maxlen2, padding='pre', truncating='pre')
			yield (batch_h, batch_b, batch_y)

def get_batch_v2(filename, tfidf_path, flag, batch_size=64, max_len_h=50, max_len_b=100):
	assert flag in ['train', 'val', 'test'], "flag should in [train, val, test], but got %s" % flag
	# load headline, body, stance from a file
	data = pickle.load(open(filename, 'rb'))
	headlines = np.array(data['h'])
	bodies = np.array(data['b'])
	y = np.array(data['y'])

	# load tfidf similarity from a file
	tfidf = pickle.load(open(tfidf_path, 'rb'))
	tfidf_sim = np.array(tfidf['%s_sim' % (flag)])

	# sorting according to the articelBodies length
	len_b = [len(v) for v in bodies]
	indices = np.argsort(len_b)

	sorted_h = headlines[indices]
	sorted_b = bodies[indices]
	sorted_y = y[indices]
	sorted_tfidf_sim = tfidf_sim[indices]

	assert len(sorted_h) == len(sorted_b) == len(sorted_y) == len(sorted_tfidf_sim)
	
	num_rows = len(sorted_h)
	batch_num = num_rows // batch_size

	# using the first 200 words in articleBodies
	for i in range(batch_num):
		batch_h = sorted_h[i*batch_size:(i+1)*batch_size]
		batch_b = sorted_b[i*batch_size:(i+1)*batch_size]
		batch_y = sorted_y[i*batch_size:(i+1)*batch_size]
		batch_tfidf_sim = sorted_tfidf_sim[i*batch_size:(i+1)*batch_size]

		# get max length of headlines and bodies in the batch
		len_batch_h = [len(v) for v in batch_h]
		len_batch_b = [len(v) for v in batch_b]
		maxlen1 = max_len_h if max_len_h < max(len_batch_h) else max(len_batch_h)
		maxlen2 = max_len_b if max_len_b < max(len_batch_b) else max(len_batch_b)

		# padding to the max length
		batch_h = pad_sequences(batch_h, maxlen1, padding='pre', truncating='post')
		batch_b = pad_sequences(batch_b, maxlen2, padding='pre', truncating='post')
		yield (batch_h, batch_b, batch_tfidf_sim, batch_y)
	
	# returning the last batch
	if num_rows / batch_size > batch_num:
		batch_h = sorted_h[batch_num*batch_size:]
		batch_b = sorted_b[batch_num*batch_size:]
		batch_y = sorted_y[batch_num*batch_size:]
		batch_tfidf_sim = sorted_tfidf_sim[batch_num*batch_size:]

		# get max length of headlines and bodies in the batch
		len_batch_h = [len(v) for v in batch_h]
		len_batch_b = [len(v) for v in batch_b]
		maxlen1 = max_len_h if max_len_h < max(len_batch_h) else max(len_batch_h)
		maxlen2 = max_len_b if max_len_b < max(len_batch_b) else max(len_batch_b)

		# padding to the max length
		batch_h = pad_sequences(batch_h, maxlen1, padding='pre', truncating='post')
		batch_b = pad_sequences(batch_b, maxlen2, padding='pre', truncating='post')
		yield (batch_h, batch_b, batch_tfidf_sim, batch_y)

def to_ids(df, word2id):
    headlines = [[w.strip().lower() for w in tokenize.word_tokenize(headline)] for headline in df['Headline'].tolist()]
    bodies = [[w.strip().lower() for w in tokenize.word_tokenize(body)] for body in df['articleBody'].tolist()]
    headlines = [[word2id[w] for w in headline if w in word2id] for headline in headlines]
    bodies = [[word2id[w] for w in body if w in word2id] for body in bodies]
    return headlines, bodies

def back_to_str(alist, id2word):
    text = [id2word.get(v) for v in alist]
    print(text)

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes).cuda()  # [D,D]
    return y[labels]            # [N,D]

def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0).cuda()
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1)).cuda()
        mask = Variable(mask, volatile=index.volatile).cuda()

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * ((1 - logit) ** self.gamma) # focal loss

        return loss.sum()/input.size(0)
