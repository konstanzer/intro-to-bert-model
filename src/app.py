import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
tf.get_logger().setLevel('ERROR')
'''
Prerequisites: pip install -q tensorflow-text
			   pip install -q tf-models-official
'''

def print_my_examples(inputs, results, user=False):
	if user:
		res = results[0][0]
		print(f'BERT scores it: {res:.3f}, ' + ranker(res))
	else:
		result_for_printing = \
			[f'{inputs[i]:<40} : score: {results[i][0]:.3f}' for i in range(len(inputs))]
		print(*result_for_printing, sep='\n')

	print()


def ranker(proba):
	if proba > .8:
		return "very positive"
	elif proba > .6:
		return "somewhat positive"
	elif proba > .4:
		return "neutral"
	elif proba > .2:
		return "somewhat negative"
	else:
		return "very negative"



if __name__ == "__main__":
	model = tf.saved_model.load('bert_imdb_model')

	examples = ['movie gave me horrible \'nam flashbacks',
				'ok not great',
				'I fell asleep during it',
				'better than Nutella',
				'so good I bought the dvd']

	results = tf.sigmoid(model(tf.constant(examples)))
	print("\n\n")
	print('Results from local tuned BERT-base model:')
	print_my_examples(examples, results)

	while True:
		print('Your review:')
		review = [input()]
		if len(review[0]) == 0:
			break
		result = tf.sigmoid(model(tf.constant(review)))
		print_my_examples(review, result, user=True)

