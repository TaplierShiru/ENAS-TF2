from typing import Dict, List, Optional, Union
import os

import tensorflow as tf
import tensorflow.keras.layers as L

from .layers import (CheckpointRestoreBase, CheckpointRestoreModule, ConvBn, FactorizedReductionLayer, 
                        StackedLayers, NASLayer, AuxHeadLayer)
from .utils import NASModelType, NASLayerType, generate_random_nas_arc, NASLayerTypeV1


NAS_LAYERS = 'nas_layers'
STEM_CONV = 'stem_conv'
FINAL_MODULE = 'final_module'
AUX_HEAD = 'aux_head'
CONFIG = 'config'


class MicroNasModel(CheckpointRestoreBase, tf.keras.Model):

    def __init__(self, 
            normal_arc: Optional[list], reduce_arc: Optional[list], 
            data_input_shape: tuple,
            arc: Union[str, List[int]] = None, 
            use_aux_heads=False,  aux_scale=0.4, num_layers=2,
            num_cells=5, out_filters=24, keep_prob=1.0,
            drop_path_keep_prob=None, global_step=None,  
            path_to_store_shared='./shared', n_classes=10,
            l2_reg=None, l1_reg=None, kernel_init=None,
            seed=None,
            **kwargs):
        super().__init__()
        self.use_aux_heads = use_aux_heads
        self.aux_scale = aux_scale
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.out_filters = out_filters
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.data_input_shape = data_input_shape
        self.n_classes = n_classes
        self.path_to_store_shared = path_to_store_shared
        os.makedirs(path_to_store_shared, exist_ok=True)

        self.l2_reg = l2_reg 
        self.l1_reg = l1_reg
        self.layer_reg = None
        if l2_reg is not None and l1_reg is not None:
            self.layer_reg = tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
        elif l2_reg is not None:
            self.layer_reg = tf.keras.regularizers.L2(l2_reg)
        elif l1_reg is not None:
            self.layer_reg = tf.keras.regularizers.L1(l1_reg)
        
        self.kernel_init = kernel_init
        if kernel_init is None:
            self.kernel_init = tf.keras.initializers.he_normal(seed=seed)

        if arc is not None:
            if isinstance(arc, str):
                arc = [int(x) for x in arc.split(" ") if x]

            if isinstance(arc, tf.Tensor):
                arc = arc.numpy().tolist()

            self.normal_arc = arc[:4 * self.num_cells]
            self.reduce_arc = arc[4 * self.num_cells:]
            self.is_fixed = True
        elif normal_arc is not None and reduce_arc is not None:
            if isinstance(normal_arc, tf.Tensor):
                normal_arc = normal_arc.numpy().tolist()

            if isinstance(reduce_arc, tf.Tensor):
                reduce_arc = reduce_arc.numpy().tolist()
            
            self.normal_arc = normal_arc
            self.reduce_arc = reduce_arc
            self.is_fixed = True
        else:
            print('MicroNasModel configured to be dynamic-builded. ')
            self.normal_arc = None
            self.reduce_arc = None
            self.is_fixed = False
        
        if self.is_fixed and len(self.normal_arc) != num_cells * 4:
            raise Exception(
                f'Size of `normal_arc`={self.normal_arc} ' 
                f'not equal to size of NAS space equal to {num_cells * 4}'
            )
        
        if self.is_fixed and len(self.reduce_arc) != num_cells * 4:
            raise Exception(
                f'Size of `reduce_arc`={self.reduce_arc} ' 
                f'not equal to size of NAS space equal to {num_cells * 4}'
            )

        if global_step is None:
            global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.global_step = global_step

        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance, 2 * pool_distance + 1]

        if self.use_aux_heads:
            self.aux_head_indices = [self.pool_layers[-1] + 1]

        # Layers
        self.stem_out_filters = out_filters * 3
        self.stem_conv = ConvBn(
            self.stem_out_filters, (3,3), 'SAME',
            start_activation=None,
            # Additional params for conv/bn
            kernel_regularizer=self.layer_reg,bias_regularizer=self.layer_reg,
            beta_regularizer=self.layer_reg,gamma_regularizer=self.layer_reg,
            kernel_initializer=self.kernel_init,
        )
        self.enas_layers_dict: Dict[str, Dict[str, CheckpointRestoreModule]] = dict()
        if keep_prob > 0 and keep_prob < 1:
            self.final_module = StackedLayers([
                L.ReLU(),
                L.GlobalAveragePooling2D(),
                L.Dropout(1.0 - keep_prob),
                L.Dense(
                    n_classes, use_bias=False,
                    # Additional params
                    kernel_regularizer=self.layer_reg,bias_regularizer=self.layer_reg,
                    kernel_initializer=self.kernel_init,
                ),
            ])
        else:
            self.final_module = StackedLayers([
                L.ReLU(),
                L.GlobalAveragePooling2D(),
                L.Dense(
                    n_classes, use_bias=False,
                    # Additional params
                    kernel_regularizer=self.layer_reg,bias_regularizer=self.layer_reg,
                    kernel_initializer=self.kernel_init,
                ),
            ])
        self._build_model()

    #@tf.function
    def call(self, x: tf.Tensor, normal_arc=None, reduce_arc=None, training=False) -> Union[tf.Tensor, List[tf.Tensor]]:
        if normal_arc is not None and reduce_arc is not None and self.is_fixed:
            raise Exception(
                'Reduce and normal arc were provided ' +\
                'to the call of the model, but model in the fixed mode.\n'
            )
        
        x = self.stem_conv(x, training=training)
        layers = [x, x]

        aux_result_tensors_list = []
        for layer_id in range(self.num_layers + 2):
            arc = normal_arc
            if layer_id in self.pool_layers:
                arc = reduce_arc
                x = self._get_layer(
                    layer_id, NASLayerType.REDUCTION_LAYER,
                )(x, training=training)
                layers = [layers[-1], x]
            x  = self._get_layer(
                layer_id, NASLayerType.NORMAL_LAYER,
            )(layers, arc=arc, training=training)
            layers = [layers[-1], x]

            if self.use_aux_heads and layer_id in self.aux_head_indices and training:
                aux_result_tensors_list += [
                    self._get_layer(
                        layer_id, NASLayerType.AUX_LAYER
                    )(x, training=training)
                ]
        x = self.final_module(x, training=training)
        return x, aux_result_tensors_list

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x_input, y_true = data

        with tf.GradientTape() as tape:
            # TODO: Sample architecture and pass to the model
            y_pred, y_aux_pred_list = self(x_input, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)
            if y_aux_pred_list is not None and len(y_aux_pred_list) > 0:
                aux_loss = 0.0
                for y_aux_head_s in y_aux_pred_list:
                    aux_loss += self.compiled_loss(y_true, y_aux_head_s, regularization_losses=self.losses)
                loss += self.aux_scale * aux_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def _build_model(self):
        reduce_arc, normal_arc = self.reduce_arc, self.normal_arc
        if not self.is_fixed:
            current_mode = NASModelType.NAS_MODE
            training = True
            if reduce_arc is None or normal_arc is None:
                # Generate fake to build model and test it
                normal_arc, reduce_arc = generate_random_nas_arc(self.num_cells)
                normal_arc = tf.convert_to_tensor(normal_arc, dtype=tf.int32)
                reduce_arc = tf.convert_to_tensor(reduce_arc, dtype=tf.int32)
        else:
            current_mode = NASModelType.FIXED_MODE
            training = False
        
        x = tf.random.uniform(shape=[1, *list(self.data_input_shape)], dtype=tf.float32)
        x = self.stem_conv(x, training=training)
        layers = [x, x]
        # building layers in the micro space
        out_filters = self.out_filters
        for layer_id in range(self.num_layers + 2):
            arc = normal_arc
            if layer_id in self.pool_layers:
                out_filters *= 2
                new_layer = FactorizedReductionLayer(out_filters)
                arc = reduce_arc
                self._set_layer(new_layer, layer_id, NASLayerType.REDUCTION_LAYER)
                x = new_layer(x, training=training)
                # TODO: Should it be `layers[0]`?
                #       See issue: https://github.com/melodyguan/enas/issues/48
                layers = [layers[-1], x]
            new_layer = NASLayer(
                # Provide arc only if current mode if FIXED
                layer_id, arc if current_mode == NASModelType.FIXED_MODE else None, 
                out_filters, self.num_cells,
                input_shape_list=[l_i.shape for l_i in layers], 
                global_step=self.global_step, mode=current_mode,
                path_to_store_shared=self.path_to_store_shared,
                layer_reg=self.layer_reg, kernel_init=self.kernel_init,
            )
            x = new_layer(
                layers, 
                # Provide arc only if current mode is NAS
                arc=arc if current_mode == NASModelType.NAS_MODE else None, 
                training=training)
            self._set_layer(
                new_layer, layer_id, 
                NASLayerType.NORMAL_LAYER
            )
            layers = [layers[-1], x]
            if self.use_aux_heads and layer_id in self.aux_head_indices and \
                    training:
                print(f'Use additional aux-head for training in layer_id={layer_id}')
                new_layer = AuxHeadLayer(layer_id, n_classes=self.n_classes)
                self._set_layer(
                    new_layer, layer_id, 
                    NASLayerType.AUX_LAYER
                )

    def _get_layer(self, layer_id: int, layer_type: NASLayerType) -> Union[NASLayer, AuxHeadLayer, StackedLayers]:
        if self.enas_layers_dict.get(str(layer_id)) is None:
            raise Exception(
                f'Layer with layer_id={layer_id} and ' 
                f'layer_type={layer_type} was not found in the model'
            )
        if self.enas_layers_dict[f'{layer_id}'].get(layer_type) is None:
            raise Exception(
                f'Layer with layer_id={layer_id} was found, but '
                f'not type layer_type={layer_type} in the model'
            )
        return self.enas_layers_dict[f'{layer_id}'][layer_type]

    def _get_layer_by_type(self, layer_type: NASLayerType) -> List[Union[NASLayer, AuxHeadLayer, StackedLayers]]:
        found_layers_list = []
        for layer_id in range(self.num_layers + 2):
            try:
                layer = self._get_layer(layer_id, layer_type)
                found_layers_list.append(layer)
            except Exception:
                continue
        return found_layers_list

    def _set_layer(self, layer: tf.Module, layer_id: int, layer_type: NASLayerType):
        if self.enas_layers_dict.get(str(layer_id)) is None:
            self.enas_layers_dict[str(layer_id)] = {}
        self.enas_layers_dict[str(layer_id)].update(
            { layer_type: layer }
        )

    def save_by_parts(
            self, path_to_save_checkpoint: str, 
            update_nas_shared_weights=True, save_nas_layers=True, 
            save_stem_conv=True, save_final_module=True, save_aux_head=True):
        assert len(self.enas_layers_dict) > 0
        if save_nas_layers or update_nas_shared_weights:
            path_to_nas_layers = f'{path_to_save_checkpoint}/{NAS_LAYERS}'
            os.makedirs(path_to_nas_layers, exist_ok=True)

            for name_level, dict_module_layer in self.enas_layers_dict.items():
                path_to_level = f'{path_to_nas_layers}/{name_level}'
                os.makedirs(path_to_level, exist_ok=True)
                for name_module, module_layer in dict_module_layer.items():
                    path_to_module_layer = f'{path_to_level}/{name_module}'
                    if save_nas_layers:
                        module_layer.save_by_parts(path_to_module_layer)
                    if update_nas_shared_weights and \
                            not self.is_fixed and isinstance(module_layer, NASLayer):
                        module_layer.update_shared()
        if save_stem_conv:
            tf.train.Checkpoint(model=self.stem_conv).write(
                f'{path_to_save_checkpoint}/{STEM_CONV}'
            )
        if save_final_module:
            tf.train.Checkpoint(model=self.final_module).write(
                f'{path_to_save_checkpoint}/{FINAL_MODULE}'
            )
        if save_aux_head:
            aux_layers = self._get_layer_by_type(NASLayerType.AUX_LAYER)
            if len(aux_layers) > 0:
                for i, aux_layer_s in enumerate(aux_layers):
                    aux_layer_s.save_by_parts(
                        f'{path_to_save_checkpoint}/{AUX_HEAD}/{i}'
                    )
        # Write normal and reduced architecture code
        with open(f'{path_to_save_checkpoint}/{CONFIG}.txt', 'w+') as fp:
            normal_arc_str = ','.join(map(str, self.normal_arc)) + '\n'
            fp.write(normal_arc_str)
            reduced_arc_str = ','.join(map(str, self.reduce_arc))
            fp.write(reduced_arc_str)
        print('Save successful!')
    
    def load_by_parts(
            self, path_to_checkpoint: str, 
            load_nas_layers=True, load_stem_conv=True, 
            load_final_module=True, load_aux_head=True):
        assert len(self.enas_layers_dict) > 0
        path_to_nas_layers = f'{path_to_checkpoint}/{NAS_LAYERS}'
        if load_nas_layers:
            for name_level, dict_module_layer in self.enas_layers_dict.items():
                for name_module, module_layer in dict_module_layer.items():
                    path_to_module_layer = f'{path_to_nas_layers}/{name_level}/{name_module}'
                    module_layer.load_by_parts(path_to_module_layer)
        if load_stem_conv:
            tf.train.Checkpoint(model=self.stem_conv).read(
                f'{path_to_checkpoint}/{STEM_CONV}'
            ).expect_partial()
        if load_final_module:
            tf.train.Checkpoint(model=self.final_module).read(
                f'{path_to_checkpoint}/{FINAL_MODULE}'
            ).expect_partial()
        if load_aux_head:
            aux_layers = self._get_layer_by_type(NASLayerType.AUX_LAYER)
            if len(aux_layers) > 0:
                for i, aux_layer_s in enumerate(aux_layers):
                    aux_layer_s.load_by_parts(path_to_checkpoint)
        print('Load successful!')

