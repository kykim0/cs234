"""Train and eval DQN agent."""

import os
import time

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from . import environments
from . import q_rnn_network

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', None,
                    'Root directory for writing logs/summaries/checkpoints.')
# Run params.
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_integer('train_steps_per_iteration', 1,
                     'Number of train steps to run per iteration.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('gradient_clipping', None, 'Norm length to clip gradients')
flags.DEFINE_integer('num_eval_episodes', 10,
                     'Number of episodes to use for eval.')
# Environment params.
flags.DEFINE_string('environment', None, 'Environment to train an agent on.')
# Replay buffer params.
flags.DEFINE_integer('replay_buffer_size', 100000, 'Size of the replay buffer.')
flags.DEFINE_integer('initial_collect_steps', 1000,
                     'Number of step to run to initialize the replay buffer.')
flags.DEFINE_integer('collect_steps_per_iteration', 1,
                     'Number of collect steps to run per iteration.')
# Network params.
flags.DEFINE_enum(
    'network_type', 'qnet', ['qnet', 'simple_rnn', 'gru', 'lstm'],
    'Type of network to use. \'qnet\' uses a Q network. Other'
    ' types use a stateful network with the corresponding RNN'
    ' cell.')
flags.DEFINE_string('fc_layers', '100', 'Fully-connected layers for Q network.')
flags.DEFINE_string('input_fc_layers', '50',
                    'Fully-connected input layers for Q RNN network.')
flags.DEFINE_string('hidden_sizes', '20', 'Number of RNN hidden units.')
flags.DEFINE_string('output_fc_layers', '20',
                    'Fully-connected output layers for Q RNN network.')
# Agent params.
flags.DEFINE_float('epsilon_greedy', None,
                   'Probability of a random action for epsilon-greedy.')
flags.DEFINE_float('boltzmann_temperature', None,
                   'Boltzmann sampling temperature.')
flags.DEFINE_integer('train_sequence_length', 1,
                     'Length of sequence of steps used for training.')
flags.DEFINE_integer('n_step_update', 1,
                     'Number of consecutive steps used for TD loss calculation.'
                     ' Only applicable to Q-network.')
flags.DEFINE_float('target_update_tau', 0.05,
                   'Weight used to gate target network update.')
flags.DEFINE_float('target_update_period', 5,
                   'Number of iterations until updating the target network.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount factor.')
flags.DEFINE_bool('use_double_q', False, 'True to use Double Q-learning.')
# Logging params.
flags.DEFINE_bool('debug_summaries', False, 'True to add debug summaries.')
flags.DEFINE_bool('summarize_grads_and_vars', False,
                  'True to add grads and vars summaries.')


def init_replay_buffer(tf_env, data_spec, train_metrics):
  """Creates and initializes a replay buffer."""
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=data_spec,
      batch_size=tf_env.batch_size,
      max_length=FLAGS.replay_buffer_size)

  # Collect initial replay data.
  logging.info(
      'Initializing replay buffer by collecting experience for %d steps with'
      ' a random policy.', FLAGS.initial_collect_steps)

  initial_collect_policy = random_tf_policy.RandomTFPolicy(
      tf_env.time_step_spec(), tf_env.action_spec())

  dynamic_step_driver.DynamicStepDriver(
      tf_env,
      initial_collect_policy,
      observers=[replay_buffer.add_batch] + train_metrics,
      num_steps=FLAGS.initial_collect_steps).run()

  return replay_buffer


def parse_str_flag(flag_value):
  """Parses a comma-separate string value."""
  return list(map(int, flag_value.split(','))) if flag_value else None


def create_agent(observation_spec, action_spec, time_step_spec, step_counter,
                 use_double_q=False):
  """Creates a DQN/DQRNN agent."""
  train_sequence_length = FLAGS.train_sequence_length
  if train_sequence_length > 1:
    q_net = q_rnn_network.RnnNetwork(
        observation_spec,
        action_spec,
        input_fc_layer_params=parse_str_flag(FLAGS.input_fc_layers),
        cell_type=FLAGS.network_type,
        hidden_size=parse_str_flag(FLAGS.hidden_sizes),
        output_fc_layer_params=parse_str_flag(FLAGS.output_fc_layers))
  else:
    q_net = q_network.QNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=parse_str_flag(FLAGS.fc_layers))
    train_sequence_length = FLAGS.n_step_update

  if FLAGS.use_double_q:
    tf_agent = dqn_agent.DdqnAgent(
        time_step_spec,
        action_spec,
        q_network=q_net,
        epsilon_greedy=FLAGS.epsilon_greedy,
        n_step_update=FLAGS.n_step_update,
        boltzmann_temperature=FLAGS.boltzmann_temperature,
        target_update_tau=FLAGS.target_update_tau,
        target_update_period=FLAGS.target_update_period,
        optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate),
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=FLAGS.gamma,
        gradient_clipping=FLAGS.gradient_clipping,
        debug_summaries=FLAGS.debug_summaries,
        summarize_grads_and_vars=FLAGS.summarize_grads_and_vars,
        train_step_counter=step_counter)
  else:
    tf_agent = dqn_agent.DqnAgent(
        time_step_spec,
        action_spec,
        q_network=q_net,
        epsilon_greedy=FLAGS.epsilon_greedy,
        n_step_update=FLAGS.n_step_update,
        boltzmann_temperature=FLAGS.boltzmann_temperature,
        target_update_tau=FLAGS.target_update_tau,
        target_update_period=FLAGS.target_update_period,
        optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate),
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=FLAGS.gamma,
        gradient_clipping=FLAGS.gradient_clipping,
        debug_summaries=FLAGS.debug_summaries,
        summarize_grads_and_vars=FLAGS.summarize_grads_and_vars,
        train_step_counter=step_counter)
  return tf_agent, train_sequence_length


def train_eval():
  """A simple train and eval for DQN."""
  logdir = FLAGS.logdir
  train_dir = os.path.join(logdir, 'train')
  eval_dir = os.path.join(logdir, 'eval')

  summary_flush_millis = 10 * 1000
  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summary_flush_millis)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summary_flush_millis)
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=FLAGS.num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=FLAGS.num_eval_episodes)
  ]

  global_step = tf.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % 1000, 0)):
    env = FLAGS.environment
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env))
    eval_tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env))

    tf_agent, train_sequence_length = create_agent(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        tf_env.time_step_spec(),
        global_step,
        FLAGS.use_double_q)
    tf_agent.initialize()

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = init_replay_buffer(
        tf_env, tf_agent.collect_data_spec, train_metrics)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=FLAGS.collect_steps_per_iteration)

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    collect_driver.run = common.function(collect_driver.run)
    tf_agent.train = common.function(tf_agent.train)

    results = metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=FLAGS.num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
    )
    metric_utils.log_metrics(eval_metrics)

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=FLAGS.batch_size,
        num_steps=train_sequence_length + 1).prefetch(3)
    iterator = iter(dataset)

    def train_step():
      experience, _ = next(iterator)
      return tf_agent.train(experience)

    train_step = common.function(train_step)

    for _ in range(FLAGS.num_iterations):
      start_time = time.time()
      time_step, policy_state = collect_driver.run(
          time_step=time_step,
          policy_state=policy_state,
      )
      for _ in range(FLAGS.train_steps_per_iteration):
        train_loss = train_step()
      time_acc += time.time() - start_time

      if global_step.numpy() % 1000 == 0:
        logging.info('step = %d, loss = %f', global_step.numpy(),
                     train_loss.loss)
        steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        tf.compat.v2.summary.scalar(name='global_steps_per_sec',
                                    data=steps_per_sec, step=global_step)
        timed_at_step = global_step.numpy()
        time_acc = 0

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2])

      if global_step.numpy() % 10000 == 0:
        train_checkpointer.save(global_step=global_step.numpy())

      if global_step.numpy() % 5000 == 0:
        policy_checkpointer.save(global_step=global_step.numpy())

      if global_step.numpy() % 20000 == 0:
        rb_checkpointer.save(global_step=global_step.numpy())

      if global_step.numpy() % 1000 == 0:
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=FLAGS.num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )
        metric_utils.log_metrics(eval_metrics)
    return train_loss


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_v2_behavior()
  train_eval()


if __name__ == '__main__':
  flags.mark_flag_as_required('logdir')
  flags.mark_flag_as_required('environment')
  app.run(main)
