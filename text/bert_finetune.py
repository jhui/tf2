# Modified from
# https://www.tensorflow.org/official_models/fine_tuning_bert

# pip install -q tf-models-official
import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from official import nlp
from official.nlp import bert
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.tokenization

# gs_folder_bert identifies the remote directory storing the configuration, vocabulary,
# and a pre-trained checkpoint for the BERT model used in this example.
gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"

# Pre-trained BERT encoder
hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"

# glue containss the downloaded 'test', 'train', 'validation' data for glue/mrpc
glue, info = tfds.load('glue/mrpc', with_info=True,
                       # It's small, load the whole dataset
                       batch_size=-1)

# Set up tokenizer to generate Tensorflow dataset
tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
    do_lower_case=True)


# Tokenize a sentence into a sequence of integers (word indexes)
# Each sentence is end with [SEP] (seperator token)
def encode_sentence(s, tokenizer):
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(glue_dict, tokenizer):
    # Sentence1 contains a batch of samples
    # and it holds the first sentences.
    sentence1 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict["sentence1"])])

    sentence2 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict["sentence2"])])

    # Prepend a [CLS] token for each sample
    # To indicate the start of a classification sample.
    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat(
        [type_cls, type_s1, type_s2], axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs


glue_train = bert_encode(glue['train'], tokenizer)
glue_train_labels = glue['train']['label']

glue_validation = bert_encode(glue['validation'], tokenizer)
glue_validation_labels = glue['validation']['label']

glue_test = bert_encode(glue['test'], tokenizer)
glue_test_labels = glue['test']['label']

bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

bert_config = bert.configs.BertConfig.from_dict(config_dict)

bert_classifier, bert_encoder = bert.bert_models.classifier_model(
    bert_config, num_labels=2)

# key = "input_word_ids", val[:10] ten training samples
glue_batch = {key: val[:10] for key, val in glue_train.items()}

bert_classifier(
    glue_batch, training=True
).numpy()

checkpoint = tf.train.Checkpoint(model=bert_encoder)
checkpoint.restore(
    os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()

# Set up epochs and steps
epochs = 3

batch_size = 32
eval_batch_size = 32

train_data_size = len(glue_train_labels)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

# creates an optimizer with learning rate schedule
optimizer = nlp.optimization.create_optimizer(
    2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)

bert_classifier.fit(
    glue_train, glue_train_labels,
    validation_data=(glue_validation, glue_validation_labels),
    batch_size=32,
    epochs=epochs)

my_examples = bert_encode(
    glue_dict={
        'sentence1': [
            'The rain in Spain falls mainly on the plain.',
            'Look I fine tuned BERT.'],
        'sentence2': [
            'It mostly rains on the flat lands of Spain.',
            'Is it working? This does not match.']
    },
    tokenizer=tokenizer)

result = bert_classifier(my_examples, training=False)

result = tf.argmax(result).numpy()
print(result)
print(np.array(info.features['label'].names)[result])

export_dir = './saved_model'
tf.saved_model.save(bert_classifier, export_dir=export_dir)

reloaded = tf.saved_model.load(export_dir)
reloaded_result = reloaded([my_examples['input_word_ids'],
                            my_examples['input_mask'],
                            my_examples['input_type_ids']], training=False)

# Re-encoding a large dataset

# Describe which features of the dataset should be transformed:
processor = nlp.data.classifier_data_lib.TfdsProcessor(
    tfds_params="dataset=glue/mrpc,text_key=sentence1,text_b_key=sentence2",
    process_text_fn=bert.tokenization.convert_to_unicode)

# Set up output of training and evaluation Tensorflow dataset
train_data_output_path = "./mrpc_train.tf_record"
eval_data_output_path = "./mrpc_eval.tf_record"

max_seq_length = 128
batch_size = 32
eval_batch_size = 32

# Generate and save training data into a tf record file
input_meta_data = (
    nlp.data.classifier_data_lib.generate_tf_record_from_data_file(
        processor=processor,
        data_dir=None,  # It is `None` because data is from tfds, not local dir.
        tokenizer=tokenizer,
        train_data_output_path=train_data_output_path,
        eval_data_output_path=eval_data_output_path,
        max_seq_length=max_seq_length))

# Create tf.data input pipelines from those TFRecord files:
training_dataset = bert.run_classifier.get_dataset_fn(
    train_data_output_path,
    max_seq_length,
    batch_size,
    is_training=True)()

evaluation_dataset = bert.run_classifier.get_dataset_fn(
    eval_data_output_path,
    max_seq_length,
    eval_batch_size,
    is_training=False)()


# ALternatively, we can use the code below to create dataset for more controls.
# Create tf.data.Dataset for training and evaluation
def create_classifier_dataset(file_path, seq_length, batch_size, is_training):
    """Creates input dataset from (tf)records files for train/eval."""
    dataset = tf.data.TFRecordDataset(file_path)
    if is_training:
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()

    def decode_record(record):
        name_to_features = {
            'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'label_ids': tf.io.FixedLenFeature([], tf.int64),
        }
        return tf.io.parse_single_example(record, name_to_features)

    def _select_data_from_record(record):
        x = {
            'input_word_ids': record['input_ids'],
            'input_mask': record['input_mask'],
            'input_type_ids': record['segment_ids']
        }
        y = record['label_ids']
        return (x, y)

    dataset = dataset.map(decode_record,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        _select_data_from_record,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


# Set up batch sizes
batch_size = 32
eval_batch_size = 32

# Return Tensorflow dataset
training_dataset = create_classifier_dataset(
    train_data_output_path,
    input_meta_data['max_seq_length'],
    batch_size,
    is_training=True)

evaluation_dataset = create_classifier_dataset(
    eval_data_output_path,
    input_meta_data['max_seq_length'],
    eval_batch_size,
    is_training=False)

# TFModels BERT on TFHub

import tensorflow_hub as hub

hub_model_name = "bert_en_uncased_L-12_H-768_A-12"
hub_encoder = hub.KerasLayer(f"https://tfhub.dev/tensorflow/{hub_model_name}/2",
                             trainable=True)

result = hub_encoder(
    inputs=[glue_train['input_word_ids'][:10],
            glue_train['input_mask'][:10],
            glue_train['input_type_ids'][:10],],
    training=False,
)

# Or a build a classifier onto the encoder from TensorFlow Hub
hub_classifier, hub_encoder = bert.bert_models.classifier_model(
    # Caution: Most of `bert_config` is ignored if you pass a hub url.
    bert_config=bert_config, hub_module_url=hub_url_bert, num_labels=2)


