'''
Created on Aug 3, 2016

@author: mkroutikov
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import codecs

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from vocab import Vocab

from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate.seq2seq_model import Seq2SeqModel
from summary_graph import summary_graph
from vose_alias_sampler import VoseAliasSampler


tf.app.flags.DEFINE_float("learning_rate",               0.5, "Learning rate (default 0.5)")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much (default 0.99)")
tf.app.flags.DEFINE_float("max_gradient_norm",           5.0, "Clip gradients to this norm (default 5.0)")
tf.app.flags.DEFINE_integer("batch_size",                 64, "Batch size to use during training (default 64)")
tf.app.flags.DEFINE_integer("size",                      256, "Size of each model layer (default 256)")
tf.app.flags.DEFINE_integer("num_layers",                  3, "Number of layers in the model (default 3)")
tf.app.flags.DEFINE_integer("vocab_size",                150, "Vocabulary size (default 150)")
tf.app.flags.DEFINE_float("dev_fraction",               0.02, "Fraction of data to use for DEV (default 0.02)")
tf.app.flags.DEFINE_string("data_dir",                "/tmp", "Data directory (default /tmp)")
tf.app.flags.DEFINE_string("train_dir",               "/tmp", "Training directory (default /tmp)")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",     1000, "How many training steps to do per checkpoint (default 200)")
tf.app.flags.DEFINE_boolean("decode",                  False, "Set to True for interactive decoding (default False)")
tf.app.flags.DEFINE_boolean("self_test",               False, "Run a self-test if this is set to True (default False)")

FLAGS = tf.app.flags.FLAGS

_buckets = [(120, 121), (200, 201)]


def read_data(source_path, vocab, dev_fraction=0.05):
    """Read data from source and target files and put into buckets.
    Args:
      source_path: path to the files with records
    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    
    train_data_set = [[] for _ in _buckets]
    dev_data_set = [[] for _ in _buckets]

    counter = 0
    with codecs.open(source_path, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            source_ids = [vocab.encode(c) for c in line]
            target_ids = source_ids + [Vocab.EOS_ID]
            
            for bucket_id, (maxlen_source, maxlen_target) in enumerate(_buckets):
                if len(source_ids) < maxlen_source and len(target_ids) < maxlen_target:
                    if random.random() < dev_fraction:
                        dev_data_set[bucket_id].append([source_ids, target_ids])
                    else:
                        train_data_set[bucket_id].append([source_ids, target_ids])
                    break

    return train_data_set, dev_data_set


def fake_data_identity(num_samples=250000, sample_len=9, vocab_size=150, dev_fraction=0.05):
    train_data_set = [[] for _ in _buckets]
    dev_data_set = [[] for _ in _buckets]
    
    for _ in range(num_samples):
        
        source_ids = [random.randint(4, vocab_size-1) for _ in range(sample_len)]
        target_ids = sorted(source_ids) + [Vocab.EOS_ID]

        for bucket_id, (maxlen_source, maxlen_target) in enumerate(_buckets):
            if len(source_ids) < maxlen_source and len(target_ids) < maxlen_target:
                if random.random() < dev_fraction:
                    dev_data_set[bucket_id].append([source_ids, target_ids])
                else:
                    train_data_set[bucket_id].append([source_ids, target_ids])
                break
    
    return train_data_set, dev_data_set


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    model = Seq2SeqModel(
        FLAGS.vocab_size, FLAGS.vocab_size, _buckets,
        FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
        forward_only=forward_only)
    
    #for var in tf.all_variables():
    #    print(var.name, var.dtype)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    
    return model


def train():
    
    if not os.path.isdir(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
        print('Created new training directory', FLAGS.train_dir)
    
    # load vocabulary
    data_filename = os.path.join(FLAGS.data_dir, 'data.txt')
    vocab_filename = os.path.join(FLAGS.data_dir, 'vocab_%d.txt' % FLAGS.vocab_size)
    if os.path.isfile(vocab_filename):
        print('Loading cached vocabulary', vocab_filename)
        vocab = Vocab.load(vocab_filename)
    else:
        print('Computing vocabulary')
        with codecs.open(data_filename) as f:
            def char_iter():
                for line in f:
                    for c in line.strip():
                        yield c
            vocab = Vocab.from_data(char_iter(), FLAGS.vocab_size)
        vocab.save(vocab_filename)
        print('Saved vocabulary', vocab_filename)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data")
    train_set, dev_set = read_data(data_filename, vocab=vocab, dev_fraction=FLAGS.dev_fraction)
    '''
    train_set, dev_set = fake_data_identity(num_samples=250000)
    '''

    train_bucket_sizes = [len(x) for x in train_set]
    bucket_sampler = VoseAliasSampler(train_bucket_sizes)

    print('Train buckets:')
    for bid in range(len(_buckets)):
        print('\t', _buckets[bid], train_bucket_sizes[bid])

    with tf.Session() as sess:
        
        ''' create two summaries: training cost and validation cost '''
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)
        summary_train = summary_graph('Training cost', ema_decay=0.95)
        summary_valid = summary_graph('Validation cost')

        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)
        
        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution.
            bucket_id = bucket_sampler()
            
            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1
            print(current_step, 'loss=%4.2f' % step_loss, (time.time() - start_time))

            sess.run([summary_train.update], {summary_train.x: loss})
            
            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity %.2f (loss=%.4f)" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity, loss))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                  sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                # Run evals on development set and print their perplexity.
                step_time, loss = 0.0, 0.0
                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("  eval: bucket %d perplexity %.2f (loss=%.3f)" % (bucket_id, eval_ppx, eval_loss))

                ''' write out summary events '''
                buffer, = sess.run([summary_train.summary])
                summary_writer.add_summary(buffer, model.global_step.eval())
                
                sess.run([summary_valid.update], {summary_valid.x: eval_loss})
                buffer, = sess.run([summary_valid.summary])
                summary_writer.add_summary(buffer, model.global_step.eval())
                
                summary_writer.flush()
                

def decode():

    vocab_filename = os.path.join(FLAGS.data_dir, 'vocab_%d.txt' % FLAGS.vocab_size)
    print('Loading cached vocabulary', vocab_filename)
    vocab = Vocab.load(vocab_filename)

    # need to load summaries to preserve var name assgnments. Should have named model vars explicitly!
    summary_train = summary_graph('Training cost', ema_decay=0.95)
    summary_valid = summary_graph('Validation cost')
    
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.
    
        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = [vocab.encode(c) for c in sentence]
            # Which bucket does it belong to?
            bucket_id = min([b for b in xrange(len(_buckets))
                               if _buckets[b][0] > len(token_ids)])
          
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
          
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            lg = [np.exp(logit) / np.sum(np.exp(logit), axis=1) for logit in output_logits]
            lg = [np.max(l, axis=1) for l in lg]
            lg = lg[:len(sentence)]
            outputs = outputs[:len(sentence)]
            # If there is an EOS symbol in outputs, cut them at that point.
            #if Vocab.EOS_ID in outputs:
            #    eos_index = outputs.index(Vocab.EOS_ID)
            #    lg = lg[:eos_index]
            
            lg_min = min(lg)
            print('Logit min:', lg_min)
            print('Logit avg:', sum(lg) / len(lg))
            
            print(''.join('v' if l==lg_min else ' ' for l in lg))

            # Print out French sentence corresponding to outputs.
            print(''.join([tf.compat.as_str(vocab.decode(output)) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
  