import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import pickle
import random
import json
nltk.download('punkt_tab')
stemmer = LancasterStemmer()

with open("intents.json", "r") as file:
	data = json.load(file)

try:
	with open('data.pickle', 'rb') as file:
		words, labels, training, output = pickle.load(file)
except Exception as e:
	print(e)
	words = []
	labels = []
	docs_x = []
	docs_y = []

	for intent in data["Ichika amasawa"]:
		for pattern in intent["patterns"]:
			tokenized_word = nltk.word_tokenize(pattern)
			words.extend(tokenized_word)
			docs_x.append(tokenized_word)
			docs_y.append(intent["tag"])

			if intent["tag"] not in labels:
				labels.append(intent["tag"])

	non_read = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
	words = [stemmer.stem(word.lower()) for word in words if word not in non_read]
	words = sorted(list(set(words)))

	labels = sorted(labels)

	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	for x, doc in enumerate(docs_x):
		bag = []
		wrds = [stemmer.stem(word.lower()) for word in doc]

		for word in words:
			if word in wrds:
				bag.append(1)
			else:
				bag.append(0)

		output_row = out_empty.copy()
		output_row[labels.index(docs_y[x])] = 1

		training.append(bag)
		output.append(output_row)

	training = np.array(training)
	output = np.array(output)
	with open('data.pickle', 'wb') as file:
		pickle.dump((words, labels, training, output), file)
print('here')
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
try:
	model.load('model.tflearn')
except Exception:
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")


def bag_of_word(s, words):
	bag = [0 for _ in range(len(words))]
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, j in enumerate(words):
			if j == se:
				bag[i] = 1

	return np.array(bag)


def chat(user_input):
	result = model.predict([bag_of_word(user_input, words)])
	result_index = np.argmax(result)
	tag = labels[result_index]
	for tg in data["Ichika amasawa"]:
		if tg["tag"] == tag:
			responses = tg["responses"]
			return random.choice(responses)

