"""Train and eval DDPG agent."""

import functools
import os
import time

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import actor_rnn_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.td3 import td3_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from . import environments

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', None,
                    'Root directory for writing logs/summaries/checkpoints.')
# Run params.
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_float('actor_learning_rate', 1e-4, 'Actor LR.')
flags.DEFINE_float('critic_learning_rate', 1e-3, 'Critic LR.')
flags.DEFINE_integer('num_eval_episodes', 10,
                     'Number of episodes to use for eval.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_integer('train_steps_per_iteration', 1,
                     'Number of train steps to run per iteration.')
# Environment params.
flags.DEFINE_string('environment', None, 'Environment to train an agent on.')
# Replay buffer params.
flags.DEFINE_integer('collect_steps_per_iteration', 1,
                     'Number of collect steps to run per iteration.')
flags.DEFINE_integer('initial_collect_steps', 1000,
                     'Number of step to run to initialize the replay buffer.')
flags.DEFINE_integer('replay_buffer_size', 1000, 'Size of the replay buffer.')
# Agent params.
flags.DEFINE_bool('use_rnn', False, 'True to use LSTM actor-critic.')
flags.DEFINE_string('actor_fc_layers', '400,300',
                    'Fully-connected layers for actor network.')
flags.DEFINE_string('actor_lstm_sizes', '40', 'LSTM layers for actor network.')
flags.DEFINE_string('actor_output_fc_layers', '100',
                    'Fully-connected layers output layers for actor network.')
flags.DEFINE_string('critic_action_fc_layers', None,
                    'Fully-connected layers for observations for critic network.')
flags.DEFINE_string('critic_joint_fc_layers', '300',
                    'Joint fully-connected layers for critic network.')
flags.DEFINE_string('critic_lstm_sizes', '40', 'LSTM layers for critic network.')
flags.DEFINE_string('critic_obs_fc_layers', '400',
                    'Fully-connected layers for observations for critic network.')
flags.DEFINE_string('critic_output_fc_layers', '100',
                    'Fully-connected layers output layers for critic network.')
flags.DEFINE_float('exploration_noise_std', 0.1, 'Gaussian exploration std.')
flags.DEFINE_float('gamma', 0.995, 'A discount factor for future rewards.')
flags.DEFINE_float('ou_damping', 0.15, 'Damping factor for the OU noise.')
flags.DEFINE_float('ou_stddev', 0.2,
                   'Std dev for OU noise added to the default collect policy.')
flags.DEFINE_float('target_update_period', 5,
                   'Number of iterations until updating the target network.')
flags.DEFINE_float('target_update_tau', 0.05,
                   'Weight used to gate target network update.')
flags.DEFINE_integer('train_sequence_length', 1,
                     'Length of sequence of steps used for training.')


def parse_str_flag(flag_value):
  """Parses a comma-separate string value."""
  return list(map(int, flag_value.split(','))) if flag_value else None


def train_eval():
  """A simple train and eval for DDPG."""
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

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % 1000, 0)):
    env = FLAGS.environment
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env))
    eval_tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env))

    if FLAGS.use_rnn:
      actor_net = actor_rnn_network.ActorRnnNetwork(
        tf_env.time_step_spec().observation,
        tf_env.action_spec(),
        input_fc_layer_params=parse_str_flag(FLAGS.actor_fc_layers),
        lstm_size=parse_str_flag(FLAGS.actor_lstm_sizes),
        output_fc_layer_params=FLAGS.actor_output_fc_layers)
      critic_net = critic_rnn_network.CriticRnnNetwork(
          input_tensor_spec=(
              tf_env.time_step_spec().observation, tf_env.action_spec()),
          observation_fc_layer_params=parse_str_flag(
              FLAGS.critic_obs_fc_layers),
          action_fc_layer_params=parse_str_flag(FLAGS.critic_action_fc_layers),
          joint_fc_layer_params=parse_str_flag(FLAGS.critic_joint_fc_layers),
          lstm_size=parse_str_flag(FLAGS.critic_lstm_sizes),
          output_fc_layer_params=parse_str_flag(FLAGS.critic_output_fc_layers))
    else:
      actor_net = actor_network.ActorNetwork(
          tf_env.time_step_spec().observation,
          tf_env.action_spec(),
          fc_layer_params=parse_str_flag(FLAGS.actor_fc_layers))
      critic_net = critic_network.CriticNetwork(
          input_tensor_spec=(
              tf_env.time_step_spec().observation, tf_env.action_spec()),
          observation_fc_layer_params=parse_str_flag(
              FLAGS.critic_obs_fc_layers),
          action_fc_layer_params=parse_str_flag(FLAGS.critic_action_fc_layers),
          joint_fc_layer_params=parse_str_flag(FLAGS.critic_joint_fc_layers))

    tf_agent = td3_agent.Td3Agent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.train.AdamOptimizer(
            learning_rate=FLAGS.actor_learning_rate),
        critic_optimizer=tf.train.AdamOptimizer(
            learning_rate=FLAGS.critic_learning_rate),
        exploration_noise_std=FLAGS.exploration_noise_std,
        target_update_tau=FLAGS.target_update_tau,
        target_update_period=FLAGS.target_update_period,
        td_errors_loss_fn=None,#tf.compat.v1.losses.huber_loss,
        gamma=FLAGS.gamma,
        train_step_counter=global_step)
    tf_agent.initialize()

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=FLAGS.replay_buffer_size)

    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env, collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=FLAGS.initial_collect_steps)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env, collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=FLAGS.collect_steps_per_iteration)

    initial_collect_driver.run = common.function(initial_collect_driver.run)
    collect_driver.run = common.function(collect_driver.run)
    tf_agent.train = common.function(tf_agent.train)
    # Collect initial replay data.
    initial_collect_driver.run()

    results = metric_utils.eager_compute(
        eval_metrics, eval_tf_env, eval_policy,
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

    # Dataset to generate trajectories.
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=FLAGS.batch_size,
        num_steps=(FLAGS.train_sequence_length + 1)).prefetch(3)
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
        tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_sec, step=global_step)
        timed_at_step = global_step.numpy()
        time_acc = 0

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2])

      if global_step.numpy() % 10000 == 0:
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
  app.run(main)
