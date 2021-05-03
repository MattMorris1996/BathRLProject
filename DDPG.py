import random
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.optimizers import Adam
from Noise import Noise

from PrioReplay import PrioReplay

class Actor(keras.Model):
    """Attempted model"""
    pass
    # @tf.function
    # def train_step(self, data):
    #     state, critic_value = data
    #     with tf.GradientTape() as tape:
    #     # actions = self.model(states, training=True)
    #     # critic_value = self.state_action_model(states_actions, training=True)
    #     #     critic_value = self.state_action_model(states_actions)
    #         actor_loss = -tf.math.reduce_mean(critic_value)
    #
    #     actor_grad = tape.gradient(actor_loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(actor_grad, self.trainable_variables))


class Critic(keras.Model):
    pass
    # @tf.function
    # def train_step(self, data):
    #     y_s, y_pred = data
    #     with tf.GradientTape() as tape:
    #         y = self(y_s, training=True)  # Forward pass
    #         # Compute the loss value
    #         # (the loss function is configured in `compile()`)
    #         loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    #
    #     critic_grad = tape.gradient(loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(critic_grad, self.trainable_variables))


class DDPG:
    """The most perfect DDPG Agent you have ever seen"""

    # Parameters taken from various sources
    epsilon = 0
    epsilon_min = 0
    decay_rate = 0.9

    learn_start = 1000
    gamma = 0.99
    alpha = 0.002
    tau = 0.005

    mem_len = 100000
    pr_replay = PrioReplay(mem_len)

    def __init__(self, env, seed):

        self.env = env
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.env.seed(seed)
        self.actor = self.create_model()
        self.target_actor = self.create_model()
        self.noise = Noise(self.env.action_space.shape[0], seed, theta=0.2, sigma=0.5)  # noise is actually OpenAI
        # baselines OU Noise wrapped in another OUNoise function

        self.critic = self.create_model((self.env.observation_space.shape[0], self.env.action_space.shape[0]))
        self.target_critic = self.create_model((self.env.observation_space.shape[0], self.env.action_space.shape[0]))
        self.target_critic.set_weights(self.critic.get_weights())  # ensure initial weights are equal for networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.reset()
        # return self.actor

    def create_model(self, input=None):
        """Generate neural network models based on inputs, defaults to Actor model"""
        last_init = tf.random_uniform_initializer(minval=-0.003,
                                                  maxval=0.003)  # To prevent actor network from causing steep gradients
        if input is None:
            input = self.env.observation_space.shape[0]  # Actor
            inputs = keras.layers.Input(shape=(input,))
            hidden = keras.layers.Dense(256, activation="relu")(inputs)
            hidden = keras.layers.Dense(256, activation="relu")(hidden)
            outputs = keras.layers.Dense(self.env.action_space.shape[0], activation="tanh",
                                         kernel_initializer=last_init)(hidden)
            model = Actor(inputs, outputs)
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.alpha / 2,
                decay_steps=1e9,
                decay_rate=1)  # This could allow us to use decaying learning rate
            model.compile(loss="huber_loss", optimizer=Adam(
                learning_rate=lr_schedule))  # Compile model with optimizer so we can apply tape.gradient later
        else:  # Critic
            input_o, input_a = input
            input1 = keras.layers.Input(shape=(input_o,))
            input2 = keras.layers.Input(shape=(input_a,))
            input11 = keras.layers.Dense(16, activation="relu")(input1)
            input11 = keras.layers.Dense(32, activation="relu")(input11)
            input21 = keras.layers.Dense(32, activation="relu")(input2)
            cat = keras.layers.Concatenate()([input11, input21])
            hidden = keras.layers.Dense(256, activation="relu")(cat)
            hidden = keras.layers.Dense(256, activation="relu")(hidden)
            outputs = keras.layers.Dense(1, activation="linear", kernel_initializer=last_init)(hidden)
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.alpha / 1,
                decay_steps=1e9,
                decay_rate=1)
            model = Critic([input1, input2], outputs)
            model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=lr_schedule))  # mean_squared_error
        return model

    @tf.function  # EagerExecution for speeeed
    def replay(self, states, actions, rewards, next_states,
               importance_weights):  # , actor, target_actor, critic, target_critic):
        """tf function that replays sampled experience to update actor and critic networks using gradient"""
        # Very much inspired by Keras tutorial: https://keras.io/examples/rl/ddpg_pendulum/

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states, training=True)
            q_target = rewards + self.gamma * self.target_critic([next_states, target_actions], training=True)
            q_current = self.critic([states, actions], training=True)

            error_sq = tf.math.square(q_target - q_current)

            critic_loss = 1 / 64 * tf.math.reduce_sum(importance_weights * error_sq)

            # critic_loss = tf.math.reduce_mean(tf.math.square(q_target - q_current))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions_prediction = self.actor(states, training=True)
            q_current = self.critic([states, actions_prediction], training=True)
            actor_loss = -tf.math.reduce_mean(q_current)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

    @tf.function
    def update_weight(self, target_weights, weights, tau):
        """tf function for updating the weights of selected target network"""
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def train_target(self):
        """Standard function to update target networks by tau"""
        self.update_weight(self.target_actor.variables, self.actor.variables, self.tau)
        self.update_weight(self.target_critic.variables, self.critic.variables, self.tau)

    def sample2batch(self, batch_size=64):
        """Return a set of Tensor samples from the memory buffer of batch_size, default is 64"""
        # batch_size = 64

        if len(self.pr_replay.buffer) < batch_size:  # return nothing if not enough experiences available
            return
        # Generate batch and emtpy arrays
        samples, importance_weights = self.pr_replay.sample(batch_size, self.target_actor, self.target_critic, self.critic, self.gamma)
        next_states = np.zeros((batch_size, self.env.observation_space.shape[0]))
        states = np.zeros((batch_size, self.env.observation_space.shape[0]))
        rewards = np.zeros((batch_size, 1))
        actions = np.zeros((batch_size, self.env.action_space.shape[0]))

        # Separate batch into arrays
        for idx, sample in enumerate(samples):
            state, action, reward, next_state, terminal = sample
            states[idx] = state
            actions[idx] = action
            rewards[idx] = reward
            next_states[idx] = next_state

        # Convert arrays to tensors so we can use replay as a callable TensorFlow graph
        states = tf.convert_to_tensor(states)
        rewards = tf.convert_to_tensor(rewards)
        rewards = tf.cast(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions)
        next_states = tf.convert_to_tensor(next_states)

        return states, actions, rewards, next_states, importance_weights

    def train(self, state, action, reward, next_state, terminal, steps):
        """Function call to update buffer and networks at predetermined intervals"""
        self.pr_replay.add([state, action, reward, next_state, terminal])  # Add new data to buffer
        if steps % 1 == 0 and len(self.pr_replay.buffer) > self.learn_start:  # Sample every X steps
            states, actions, rewards, next_states, importance_weights  = self.sample2batch()
            self.replay(states, actions, rewards, next_states, importance_weights)
        if steps % 1 == 0:  # Update targets only every X steps
            self.train_target()

    def reset(self):
        self.epsilon *= self.decay_rate
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def choose_action(self, state, scale=False):
        """Choose action based on policy and noise function. Scale option used to limit maximum actions"""
        # self.epsilon *= self.decay
        # self.epsilon = round(max(self.epsilon / 1000, self.epsilon), 5)
        # print(state[0])
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)  # convert to tensor for speeeed
        if np.random.random() < self.epsilon:  # If using epsilon instead of exploration noise
            return random.uniform(-1, 1)
        if scale:
            return np.clip(0.33 * (self.env.action_space.sample()) + self.noise.sample(), -1, 1)
        return np.clip(1 * tf.squeeze(self.actor(state)).numpy() + self.noise.sample(), -1,
                       1)  # np.argmax(self.model.predict(state))  # action
