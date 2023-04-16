import os
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L

from .micro_nas_model import MicroNasModel
from .micro_lstm import MicroLSTM
from .utils import ClipGradsMode


CONTROLLER_FOLDER = 'controller'
MICRO_ENAS_MODEL_FOLDER = 'micro_enas_model'


class MicroChild:

    def __init__(self,
               input_shape: tuple,
               nas_controller: MicroLSTM,
               opt_nas: tf.keras.optimizers.Optimizer,
               opt_controller: tf.keras.optimizers.Optimizer,
               use_aux_heads=False, fixed_arc=None, num_layers=2,
               num_cells=5, out_filters=24, keep_prob=1.0,
               drop_path_keep_prob=None,
               l2_reg=None, l1_reg=None, kernel_init=None,
               clip_mode: ClipGradsMode=None, grad_bound_nas_model=5.0,
               save_path_folder: str = './weights', 
               name="child", **kwargs):
        super().__init__()
        self.input_shape = input_shape
        self.nas_controller = nas_controller
        self.opt_nas = opt_nas
        self.opt_controller = opt_controller
        
        self.use_aux_heads = use_aux_heads
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        
        # TODO: Does not work l1/l2 as expected... Layers and modules must be part of the Keras?
        #       Should we avaiod Keras model here and calculate all by hand?
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.kernel_init = kernel_init

        self.grad_bound_nas_model = grad_bound_nas_model
        self.clip_mode = clip_mode

        self.out_filters = out_filters
        self.num_layers = num_layers
        self.num_cells = num_cells
        # TODO: Fixed is impossible here to train
        #       Write function to train only fixed?
        #       Remove this api?
        self.fixed_arc = fixed_arc
        self.name = name

        self.enas_weights_save_path = f'{save_path_folder}/{MICRO_ENAS_MODEL_FOLDER}'
        self.controller_weights_save_path = f'{save_path_folder}/{CONTROLLER_FOLDER}' 
        self.save_path_folder = save_path_folder
        os.makedirs(self.save_path_folder, exist_ok=True)
        os.makedirs(self.enas_weights_save_path, exist_ok=True)
        os.makedirs(self.controller_weights_save_path, exist_ok=True)

        self.model = None
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="global_step"
        )
        self.model = MicroNasModel(
            normal_arc=None, reduce_arc=None, data_input_shape=self.input_shape, 
            arc=None, use_aux_heads=self.use_aux_heads, num_layers=self.num_layers, 
            num_cells=self.num_cells, out_filters=self.out_filters, keep_prob=self.keep_prob, 
            drop_path_keep_prob=self.drop_path_keep_prob, global_step=self.global_step,
            path_to_store_shared=self.save_path_folder,
            l1_reg=self.l1_reg, l2_reg=self.l2_reg, kernel_init=self.kernel_init,
        )
        self.model.compile(self.opt_nas, loss=self.loss, metrics='accuracy', run_eagerly=False)

    #@tf.function
    def single_step_nas_micro(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x_input, y_true = data
        normal_arc, reduce_arc = None, None
        with tf.GradientTape() as tape:
            if self.fixed_arc is None:
                (normal_arc, reduce_arc), _, _, _, _ = self.nas_controller.calculate_entropy()
            else:
                # TODO: Parse fixed arc?
                pass
            y_pred, y_aux_pred_list = self.model(
                x_input, 
                normal_arc=normal_arc, reduce_arc=reduce_arc, 
                training=True
            )  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.model.compiled_loss(y_true, y_pred, regularization_losses=self.model.losses)
            if y_aux_pred_list is not None and len(y_aux_pred_list) > 0:
                aux_loss = 0.0
                for y_aux_head_s in y_aux_pred_list:
                    # TODO: Add scale as model parameter
                    aux_loss += self.model.compiled_loss(y_true, y_aux_head_s) * 0.4
                loss += self.model.aux_scale * aux_loss

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Remove gradients which are None aka not used with current normal/reduce arc
        checked_gradients = []
        checked_trainable_vars = []
        for gr, tr_var in zip(gradients, self.model.trainable_variables):
            # TODO: Is `gr` also None then using tf.function for layers? Need to test it!
            if gr is not None:
                if self.clip_mode is not None and self.clip_mode == ClipGradsMode.NORM:
                    assert self.grad_bound_nas_model, "Need grad_bound to clip gradients."
                    # TODO: Is this if-else needed?
                    if isinstance(gr, tf.IndexedSlices):
                        c_g = tf.clip_by_norm(gr.values, self.grad_bound_nas_model)
                        c_g = tf.IndexedSlices(gr.indices, c_g, dense_shape=tf.shape(tr_var))
                    else:
                        c_g = tf.clip_by_norm(gr, self.grad_bound_nas_model)
                    gr = c_g
                checked_gradients.append(gr)
                checked_trainable_vars.append(tr_var)
        # TODO: Test it!
        if self.clip_mode is not None:
            assert self.grad_bound_nas_model, "Need grad_bound to clip gradients."
            if self.clip_mode == ClipGradsMode.GLOBAL:
                checked_gradients, _ = tf.clip_by_global_norm(checked_gradients, self.grad_bound_nas_model)
            elif self.clip_mode == ClipGradsMode.NORM:
                # Applied above
                pass
            else:
                raise NotImplementedError("Unknown clip_mode {}".format(self.clip_mode))
        # Update weights
        self.model.optimizer.apply_gradients(zip(checked_gradients, checked_trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.model.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.model.metrics}

    def single_micro_eval(self, data, normal_arc=None, reduce_arc=None):
        x_input, y_true = data
        
        if self.fixed_arc is None and (normal_arc is None or reduce_arc is None):
            (normal_arc, reduce_arc), _, _, _, _ = self.nas_controller.calculate_entropy()
        elif self.fixed_arc:
            # TODO: Parse fixed arc?
            raise Exception('Option not implemented!')

        predictions, _ = self.model(
            x_input, 
            normal_arc=normal_arc, reduce_arc=reduce_arc, 
            training=False
        )  # Forward pass
        predictions = tf.argmax(predictions, axis=-1, output_type=tf.int32)
        acc = tf.equal(tf.cast(predictions, dtype=tf.int32), tf.cast(y_true, dtype=tf.int32))
        acc = tf.cast(acc, dtype=tf.float32)
        acc = tf.reduce_mean(acc)
        return acc

    def nas_micro_eval(self, test_generator: tf.data.Dataset, normal_arc=None, reduce_arc=None, hide_progress=False):
        acc_test_list = []
        if hide_progress:
            iterator = test_generator
        else:
            iterator = tqdm(test_generator)
        try:
            for data in iterator:
                acc_test_list.append(
                    self.single_micro_eval(data, normal_arc, reduce_arc)
                )
            acc_mean = tf.reduce_mean(acc_test_list)
            return acc_mean
        finally:
            if not hide_progress:
                iterator.close()

    def fit_nas(
            self, data_generator: tf.data.Dataset,
            epochs: int, steps_per_epoch: int, 
            print_period: int = 50, decay = 0.9):
        if self.model is None:
            raise Exception('In the MicroChild `fit_nas` was called, but model is not builded.')
        metric_global_dict = {}
        metric_decay_dict = {}
        for epoch_i in range(epochs):
            # Because using `enumerate` - tqdm can't take `total` by itself - so provide it as it is now
            iterator = tqdm(enumerate(data_generator.take(steps_per_epoch)), total=steps_per_epoch)
            try:
                for i, data in iterator:
                    metric_dict = self.single_step_nas_micro(data)
                    for k,v in metric_dict.items():
                        if metric_global_dict.get(k) is None:
                            metric_global_dict[k] = []
                        metric_global_dict[k] += [v]

                        if metric_decay_dict.get(k) is None:
                            metric_decay_dict[k] = v
                        else:
                            metric_decay_dict[k] = v * decay + (1 - decay) * metric_decay_dict[k]
                    
                    if (i+1) % print_period == 0:
                        str_metric = f'It: {(i+1)}/{steps_per_epoch+1} Metrics: '
                        for k,v in metric_decay_dict.items():
                            str_metric += f'{k}={round(v.numpy(), 3)} '
                        print(str_metric)

                metric_avg_dict = dict([
                    (k, sum(v)/len(v))
                    for k, v in metric_global_dict.items()
                ])

                str_metric = f'{epochs}/{epoch_i+1} Avg Metrics: '
                for k,v in metric_avg_dict.items():
                    str_metric += f'{k}={round(v.numpy(), 3)} '
                print(str_metric)
            finally:
                iterator.close()

    def single_step_nas_controller(self, x_data, y_data):
        with tf.GradientTape() as tape:
            res_data = self.nas_controller.calculate_entropy(training=True)
            (normal_arc, reduce_arc), actions_prob, actions_log_prob, samples_entropy, samples_log_prob = res_data
            predictions, _ = self.model(
                x_data, 
                normal_arc=normal_arc, reduce_arc=reduce_arc, 
                training=False # TODO: False is right here?
            )
            predictions = tf.argmax(predictions, axis=-1, output_type=tf.int32)
            reward = tf.equal(y_data, predictions)
            reward = tf.reduce_mean(tf.cast(reward, dtype=tf.float32))
            loss, reward = self.nas_controller.calculate_loss_old(
                samples_entropy, samples_log_prob,
                #actions_prob, actions_log_prob, 
                reward
            )
        grads = tape.gradient(loss, self.nas_controller.trainable_variables)
        self.opt_controller.apply_gradients(zip(grads, self.nas_controller.trainable_variables))
        return loss, reward

    def save_nas_controller(self, current_epoch: int, additional_str="saved_model_controller"):
        tf.saved_model.save(self.nas_controller, f'{self.controller_weights_save_path}/{additional_str}_{current_epoch}')
    
    def fit_nas_controller(
            self, data_generator: tf.data.Dataset, iterations: int, 
            print_period: int = 10, decay = 0.9):
        total_loss_list = []
        loss_smooth = None
        total_reward_list = []
        reward_smooth = None

        data_generator = iter(data_generator.repeat())
        iterator = tqdm(range(iterations))
        try:
            for iteration in iterator:
                x_data, y_data = next(data_generator)
                loss, reward = self.single_step_nas_controller(x_data, y_data)

                if loss_smooth is None:
                    loss_smooth = loss
                else:
                    loss_smooth = loss * decay + (1 - decay) * loss_smooth
                
                if reward_smooth is None:
                    reward_smooth = reward
                else:
                    reward_smooth = reward * decay + (1 - decay) * reward_smooth


                total_loss_list.append(loss)
                total_reward_list.append(reward)
                if iteration % print_period == 0:
                    print(
                        f'Controller loss = {float(loss_smooth.numpy())}, reward = {float(reward_smooth.numpy())}'
                    )
        finally:
            iterator.close()
        return total_loss_list, total_reward_list

    def generate_and_print_arcs(self, data_generator: tf.data.Dataset, number_to_generate=10):
        data_generator = iter(data_generator.repeat())
        for i in range(number_to_generate):
            (normal_arc, reduce_arc), _, _, _, _ = self.nas_controller.calculate_entropy()
            acc = self.single_micro_eval(
                next(data_generator), 
                normal_arc=normal_arc, reduce_arc=reduce_arc
            )

            print('=' * 37, str(i+1).zfill(2), '/', str(number_to_generate).zfill(2), '=' * 38)
            print('Normal arc=', normal_arc.numpy().tolist())
            print('Reduce arc=', reduce_arc.numpy().tolist())
            print('Accuracy=', round(acc.numpy(), 4))
            print('=' * 80)

    def fit_controller(
            self, data_generator: tf.data.Dataset, test_data_generator: tf.data.Dataset, 
            steps_per_epoch_for_model: int, num_epochs_for_model: int, 
            num_epoch_for_controller: int, epochs=50, eval_period=5):
        for epoch_i in range(epochs):
            print(f'Epoch={epoch_i+1}/{epochs}')
            # Train model
            print('Train nas-model...') 
            self.fit_nas(data_generator, num_epochs_for_model, steps_per_epoch_for_model)
            # Train nas controller
            print('Train nas-controller...')
            self.fit_nas_controller(test_data_generator, iterations=num_epoch_for_controller)
            print('Test enas-model...')
            test_acc_mean = self.nas_micro_eval(test_data_generator)
            print(f'Test accuracy={round(test_acc_mean.numpy(), 3)}')
            print('Generate architectures...')
            self.generate_and_print_arcs(test_data_generator)
            if epoch_i != 0 and (epoch_i+1) % eval_period == 0:
                self.save_nas_controller(epoch_i)
        self.save_nas_controller(epoch_i, 'saved_model_controller_final')
