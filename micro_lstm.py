import tensorflow as tf
import tensorflow.keras.layers as L
import numpy as np

from .layers import StackedRNN


class MicroLSTM(tf.Module):

    def __init__(
            self, num_branches=6,  
            lstm_size=32, num_cells=6, 
            decay=0.99,
            temperature: float = None, tanh_constant: float = None, 
            op_tanh_reduce: float = 1.0, entropy_weight: float = None):
        super().__init__()
        self.num_branches = num_branches
        self.lstm_size = lstm_size
        self.num_generated_cells = 2 # Always 2 for current implementation
        self.num_cells = num_cells
        self.decay = decay
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce
        self.entropy_weight = entropy_weight 
        
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        self.lstm = StackedRNN(
            [ 
                L.LSTM(
                    lstm_size, return_sequences=True, return_state=True,
                    kernel_initializer=initializer,
                    recurrent_initializer=initializer,
                    bias_initializer=initializer,
                )
                # TODO: Should it be equal to 2? or 1?
                for _ in range(1)
            ],
        )

        self.g_emb = tf.Variable(initializer(shape=(1, lstm_size), dtype=tf.float32), name='g-emb')
        self.w_emb = tf.Variable(initializer(shape=(num_branches, lstm_size), dtype=tf.float32), name='w-emb')
        self.w_soft = tf.Variable(initializer(shape=(lstm_size, num_branches), dtype=tf.float32), name='w-soft')
        self.b_soft = tf.Variable(
            np.array([10.0, 10.0] + [0.0] * (num_branches - 2), dtype=np.float32).reshape(1, num_branches),
            shape=[1, num_branches],
            dtype=tf.float32, name='b-soft'
        )

        b_soft_not_learned = np.array(
            [0.25, 0.25] + [-0.25] * (self.num_branches - 2), 
            dtype=np.float32
        )
        b_soft_not_learned = np.reshape(b_soft_not_learned, [1, self.num_branches])
        self.b_soft_not_learned = tf.constant(b_soft_not_learned, dtype=tf.float32)
        
        self.w_attn_1 = tf.Variable(initializer(shape=(lstm_size, lstm_size), dtype=tf.float32), name='att-w-1')
        self.w_attn_2 = tf.Variable(initializer(shape=(lstm_size, lstm_size), dtype=tf.float32), name='att-w-2')
        self.v_attn = tf.Variable(initializer(shape=(lstm_size, 1), dtype=tf.float32), name='att-v')

        self.baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.skip_rate = tf.constant(0.0, dtype=tf.float32)

    def call_lstm(self, x, prev_h=None, prev_c=None, training=False):
        if prev_h is None or prev_c is None:
            return self.lstm(
                x, training=training
            )
        return self.lstm(
            x, initial_state=[prev_h, prev_c], training=training
        )

    @tf.function
    def sample(self, prev_c=None, prev_h=None, use_bias=False, training=False):
        # TODO: Rewrite it to
        #           https://github.com/google-research/google-research/blob/698c1a53af550cf29becdb08e1c1b2f0a507dd46/enas_lm/src/controller.py#L116
        # Good example on torch:
        #           https://github.com/MengTianjian/enas-pytorch/blob/master/micro_controller.py
        arc_seq = []
        sampled_entropy_list, sampled_log_prob_list = [], []
        action_prob_list, action_log_prob_list = [], []
        
        all_h = [tf.zeros([1, self.lstm_size], dtype=tf.float32)]
        all_h_w = [tf.zeros([1, self.lstm_size], dtype=tf.float32)]
        inputs = tf.expand_dims(self.g_emb, axis=0)
        for layer_id in range(self.num_generated_cells):
            _, next_h, next_c = self.call_lstm(
                inputs, prev_h=prev_h, prev_c=prev_c, training=training
            )
            prev_h, prev_c = next_h, next_c
            all_h.append(tf.zeros_like(next_h)) # next_h or zeros with shape as next_h ???
            all_h_w.append(
                tf.matmul(next_h, self.w_attn_1)
            )
        
        layer_id = 2
        while layer_id < (self.num_cells + 2):
            arc_seq_cell = [None] * 4
            prev_layers = []
            for cell_i in range(2):
                _, next_h, next_c = self.call_lstm(
                    inputs, prev_h=prev_h, prev_c=prev_c, training=training
                )
                prev_h, prev_c = next_h, next_c
                query = tf.tanh(
                    tf.concat(all_h_w[:layer_id], axis=0) + tf.matmul(next_h, self.w_attn_2)
                )
                logits = tf.matmul(query, self.v_attn)
                logits = tf.reshape(logits, [1, layer_id])
                if self.temperature is not None:
                    logits /= self.temperature
                if self.tanh_constant is not None:
                    logits = self.tanh_constant * tf.tanh(logits)

                skip_index = tf.random.categorical(logits, 1)
                skip_index = tf.cast(skip_index, dtype=tf.int32)
                skip_index = tf.reshape(skip_index, [])
                arc_seq_cell[cell_i * 2] = skip_index
                
                action_prob = tf.nn.softmax(logits)
                action_log_prob = tf.nn.log_softmax(logits)
                action_prob_list.append(tf.gather(action_prob[0], skip_index))
                action_log_prob_list.append(tf.gather(action_log_prob[0], skip_index))

                # TODO: Its named as log-prob, but its a calculated entropy after softmax - is it right?
                log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=tf.reshape(skip_index, [1])
                )
                sampled_log_prob_list.append(log_prob)

                entropy = tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=action_prob
                ))
                sampled_entropy_list.append(entropy)

                prev_layers.append(tf.gather(all_h, skip_index))
                inputs = tf.expand_dims(prev_layers[-1], axis=0)

            for op_i in range(2):
                # Operation
                _, next_h, next_c = self.call_lstm(
                   inputs, prev_h=prev_h, prev_c=prev_c, training=training
                )
                prev_h, prev_c = next_h, next_c
                logits = tf.matmul(next_h, self.w_soft) + self.b_soft
                if self.temperature is not None:
                    logits /= self.temperature
                if self.tanh_constant is not None:
                    logits = (self.tanh_constant / self.op_tanh_reduce) * tf.tanh(logits)
                if use_bias:
                    logits += self.b_soft_not_learned
                op_id = tf.random.categorical(logits, 1)
                op_id = tf.cast(op_id, dtype=tf.int32)
                op_id = tf.reshape(op_id, [])
                arc_seq_cell[op_i * 2 + 1] = op_id
                
                action_prob = tf.nn.softmax(logits)
                action_log_prob = tf.nn.log_softmax(logits)
                action_prob_list.append(tf.gather(action_prob[0], skip_index))
                action_log_prob_list.append(tf.gather(action_log_prob[0], skip_index))

                # TODO: Its named as log-prob, but its a calculated entropy after softmax - is it right?
                log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=tf.reshape(op_id, [1])
                )
                sampled_log_prob_list.append(log_prob)

                entropy = tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=action_prob
                ))
                sampled_entropy_list.append(entropy)

                inputs = tf.expand_dims(tf.expand_dims(
                    tf.nn.embedding_lookup(self.w_emb, op_id), 
                    axis=0 ), axis=0
                )            
            _, next_c, next_h = self.call_lstm(
                inputs, prev_h=prev_h, prev_c=prev_c, training=training
            )
            arc_seq += arc_seq_cell
            all_h.append(next_h)
            all_h_w.append(tf.matmul(next_h, self.w_attn_1))
            inputs = tf.expand_dims(self.g_emb, axis=0)
            layer_id += 1
        
        arc_seq = tf.squeeze(tf.stack(arc_seq, axis=0))

        actions_prob = tf.stack(action_prob_list, axis=0)    
        actions_log_prob = tf.stack(action_log_prob_list, axis=0)

        samples_entropy = tf.concat(sampled_entropy_list, axis=0)
        samples_log_prob = tf.concat(sampled_log_prob_list, axis=0)
        return (
            arc_seq, 
            actions_prob, actions_log_prob, 
            samples_entropy, samples_log_prob, 
            prev_c, prev_h
        )
    
    @tf.function
    def calculate_entropy(self, training=False):
        arc_seq_1, actions_prob_1, action_log_prob_1, entropy_1, log_prob_1, c, h = self.sample(use_bias=True, training=training)
        arc_seq_2, actions_prob_2, action_log_prob_2, entropy_2, log_prob_2, _, _ = self.sample(prev_c=c, prev_h=h, training=training)
        sample_arc = (arc_seq_1, arc_seq_2)
        actions_prob = actions_prob_1 + actions_prob_2
        actions_log_prob = action_log_prob_1 + action_log_prob_2
        samples_entropy = entropy_1 + entropy_2
        samples_log_prob = log_prob_1 + log_prob_2
        return sample_arc, actions_prob, actions_log_prob, samples_entropy, samples_log_prob

    @tf.function
    def calculate_loss_old(self, samples_entropy, samples_log_prob, reward):
        # TODO: Original its put here, but I think its not right?
        # Additional impl:
        #               https://github.com/MarSaKi/nasnet/blob/0c6e3ccb9e89e3859b9eff582eb6c5247abc0ebb/policy_gradient.py#L97
        if self.entropy_weight is not None:
            reward += self.entropy_weight * tf.reduce_sum(samples_entropy)

        log_prob_sum = tf.reduce_sum(samples_log_prob)
        # TODO: Rewrite this part. Here its like decayded loss
        self.baseline.assign_sub((1 - self.decay) * (self.baseline - reward))
        loss = log_prob_sum * (reward - self.baseline)
        return loss, reward

    @tf.function
    def calculate_loss(self, actions_prob, actions_log_prob, reward):
        # TODO: Test it
        self.baseline.assign_sub((1 - self.decay) * (self.baseline - reward))
        reward_gain = reward - self.baseline

        policy_loss = -1 * tf.reduce_sum(actions_log_prob * reward_gain)
        entropy = -1 * tf.reduce_sum(actions_prob * actions_log_prob)
        entropy_bonus = -1 * entropy

        if self.entropy_weight is not None:
            entropy_bonus *= self.entropy_weight
        
        loss = policy_loss + entropy_bonus
        return loss, reward

    #@tf.function
    def calculate_PRO_loss(self, actions_prob, actions_log_prob, reward, clip_eps=0.2):
        # TODO: Test it
        # Good article about loss:
        #               https://huggingface.co/blog/deep-rl-ppo
        # Taken from here:
        #               https://github.com/MarSaKi/nasnet/blob/0c6e3ccb9e89e3859b9eff582eb6c5247abc0ebb/PPO.py#L149
        lower_bound = tf.ones_like(actions_prob) * (1 - clip_eps)
        upper_bound = tf.ones_like(actions_prob) * (1 + clip_eps)

        actions_prob_clipped = tf.minimum(actions_prob, upper_bound)
        actions_prob_clipped = tf.maximum(actions_prob_clipped, lower_bound)

        self.baseline.assign_sub((1 - self.decay) * (self.baseline - reward))
        reward_gain = (reward - self.baseline)
        actions_reward_prob_gain = actions_prob * reward_gain
        actions_reward_clipped_prob_gain = actions_prob_clipped * reward_gain

        actions_reward = tf.minimum(actions_reward_prob_gain, actions_reward_clipped_prob_gain)
        policy_loss = -1 * tf.reduce_sum(actions_reward)
        entropy = -1 * tf.reduce_sum(actions_prob * actions_log_prob)
        entropy_bonus = -1 * entropy
        if self.entropy_weight is not None:
            entropy_bonus *= self.entropy_weight

        return policy_loss + entropy_bonus

