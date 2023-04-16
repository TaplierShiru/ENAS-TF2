from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow.keras.layers as L

from typing import List, Optional, Union

from .utils import apply_drop_path, NASModelType, NASCellTypeV1



class CheckpointRestoreBase(ABC):

    @abstractmethod
    def save_by_parts(self, path_to_save_checkpoint: str):
        pass
    
    @abstractmethod
    def load_by_parts(self, path_to_checkpoint: str):
        pass


class CheckpointRestoreModule(CheckpointRestoreBase, tf.Module, ABC):

    def save_by_parts(self, path_to_save_checkpoint: str):
        # Just save full module to checkpoint file!
        tf.train.Checkpoint(self).write(path_to_save_checkpoint)

    def load_by_parts(self, path_to_checkpoint: str):
        # Just load full module from checkpoint file!
        tf.train.Checkpoint(self).read(path_to_checkpoint).expect_partial()


class ConvBn(tf.Module):

    def __init__(self, out_filters: int, filter_size: Union[tuple, int], padding='SAME',
            start_activation=L.ReLU(), end_activation=None,
            curr_cell: int = None, prev_cell: int = None, 
            conv_id: int = None, num_possible_inputs: int = None,
            type_op: int = None, **kwargs):
        super().__init__()

        self.out_filters = out_filters
        if isinstance(filter_size, int):
            filter_size = (filter_size, filter_size)
        self.filter_size = filter_size
        self.padding = padding

        self.start_activation = start_activation
        self.end_activation = end_activation

        self.curr_cell = curr_cell
        self.prev_cell = prev_cell
        self.conv_id = conv_id
        self.num_possible_inputs = num_possible_inputs
        self.type_op = type_op

        self.conv = L.Conv2D(
            out_filters, filter_size, padding=padding, use_bias=False,
            kernel_initializer=kwargs.get('kernel_initializer') if kwargs.get('kernel_initializer') is None else 'glorot_uniform',
            bias_initializer=kwargs.get('bias_initializer') if kwargs.get('bias_initializer') is None else 'zeros',
            kernel_regularizer=kwargs.get('kernel_regularizer'),
            bias_regularizer=kwargs.get('bias_regularizer'),
        )
        self.bn = L.BatchNormalization(
            beta_regularizer=kwargs.get('beta_regularizer'),
            gamma_regularizer=kwargs.get('gamma_regularizer'),
        )
    
    @tf.function
    def __call__(self, x, training=False):
        if self.start_activation is not None:
            x = self.start_activation(x, training=training)
        x = self.conv(x, training=training)
        x = self.bn(x, training=training)
        if self.end_activation is not None:
            x = self.end_activation(x, training=training)
        return x


class DynamicInputConvBn(tf.Module):

    def __init__(
            self, max_input_number: int, in_f_per_one: int, 
            out_filters: int, filter_size: Union[tuple, int], padding='SAME',
            start_activation=L.ReLU(), end_activation=None,
            curr_cell: int = None, prev_cell: int = None, 
            conv_id: int = None, num_possible_inputs: int = None,
            type_op: int = None, seed=None, **kwargs):
        # TODO: How to add reg (l1/l2 losses) ? Make this class as keras-layer?
        self.out_filters = out_filters
        if isinstance(filter_size, int):
            filter_size = (filter_size, filter_size)
        self.filter_size = filter_size
        self.padding = padding

        self.start_activation = start_activation
        self.end_activation = end_activation

        self.curr_cell = curr_cell
        self.prev_cell = prev_cell
        self.conv_id = conv_id
        self.num_possible_inputs = num_possible_inputs
        self.type_op = type_op

        he_normal = tf.keras.initializers.he_normal(seed=seed)
        w = he_normal((max_input_number, filter_size[0] * filter_size[1] * out_filters * in_f_per_one))
        self.w = tf.Variable(w, trainable=True, name='dynamic_inputs_conv/kernel')
        self.bn = L.BatchNormalization(
            beta_regularizer=kwargs.get('beta_regularizer'),
            gamma_regularizer=kwargs.get('gamma_regularizer'),
            name='dynamic_inputs_bn'
        )


    @tf.function
    def __call__(self, x, indices: tf.Tensor = None, training=False):
        if self.start_activation is not None:
            x = self.start_activation(x, training=training)

        if indices is None:
            w = self.w
        else:
            w = tf.gather(self.w, indices, axis=0)
        w = tf.reshape(
            w, 
            (self.filter_size[0], self.filter_size[1], -1 , self.out_filters)
        )
        x = tf.nn.conv2d(
            x, w, strides=1, padding=self.padding
        )
        x = self.bn(x, training=training)
        if self.end_activation is not None:
            x = self.end_activation(x, training=training)
        return x


class SeparableConvBn(tf.Module):

    def __init__(self, out_filters: int, filter_size: Union[tuple, int], padding: str, 
            curr_cell: int, prev_cell: int, 
            conv_id: int, num_possible_inputs: int, **kwargs):
        super().__init__()

        self.out_filters = out_filters
        if isinstance(filter_size, int):
            filter_size = (filter_size, filter_size)
        self.filter_size = filter_size
        self.padding = padding

        self.curr_cell = curr_cell
        self.prev_cell = prev_cell
        self.conv_id = conv_id
        self.num_possible_inputs = num_possible_inputs

        self.sep_conv = L.SeparableConv2D(
            out_filters, filter_size, padding=padding, use_bias=False,
            depthwise_initializer=kwargs.get('kernel_initializer') if kwargs.get('kernel_initializer') is None else 'glorot_uniform',
            pointwise_initializer=kwargs.get('kernel_initializer') if kwargs.get('kernel_initializer') is None else 'glorot_uniform',
            bias_initializer=kwargs.get('bias_initializer') if kwargs.get('bias_initializer') is None else 'zeros',
            depthwise_regularizer=kwargs.get('kernel_regularizer'),
            pointwise_regularizer=kwargs.get('kernel_regularizer'),
            bias_regularizer=kwargs.get('bias_regularizer'),
        )
        self.bn = L.BatchNormalization(
            beta_regularizer=kwargs.get('beta_regularizer'),
            gamma_regularizer=kwargs.get('gamma_regularizer'),
        )
    
    @tf.function
    def __call__(self, x, training=False):
        x = self.sep_conv(x, training=training)
        return self.bn(x, training=training)


class FactorizedReductionLayer(CheckpointRestoreModule):

    def __init__(self, out_filters: int, stride=2, **kwargs):
        super().__init__()
        self.path1_avg_pool = L.AveragePooling2D((1, 1), (stride, stride), padding='VALID')
        self.path1_conv = L.Conv2D(
            out_filters//2, (1,1), padding='VALID', use_bias=False,
            kernel_initializer=kwargs.get('kernel_initializer') if kwargs.get('kernel_initializer') is None else 'glorot_uniform',
            bias_initializer=kwargs.get('bias_initializer') if kwargs.get('bias_initializer') is None else 'zeros',
            kernel_regularizer=kwargs.get('kernel_regularizer'),
            bias_regularizer=kwargs.get('bias_regularizer'),
        )

        self.path2_pad = L.ZeroPadding2D(((0, 1), (0, 1)))
        self.path2_avg_pool = L.AveragePooling2D((1, 1), (stride, stride), padding='VALID')
        self.path2_conv = L.Conv2D(
            out_filters//2, (1,1), padding='VALID', use_bias=False,
            kernel_initializer=kwargs.get('kernel_initializer') if kwargs.get('kernel_initializer') is None else 'glorot_uniform',
            bias_initializer=kwargs.get('bias_initializer') if kwargs.get('bias_initializer') is None else 'zeros',
            kernel_regularizer=kwargs.get('kernel_regularizer'),
            bias_regularizer=kwargs.get('bias_regularizer'),
        )

        self.bn = L.BatchNormalization(
            beta_regularizer=kwargs.get('beta_regularizer'),
            gamma_regularizer=kwargs.get('gamma_regularizer'),
        )
    
    @tf.function
    def __call__(self, x, training=False):
        x1 = self.path1_avg_pool(x, training=training)
        x1 = self.path1_conv(x1, training=training)

        x2 = self.path2_pad(x, training=training)
        # TODO: Why they do that?
        # https://github.com/melodyguan/enas/blob/4bcfd73b524627ea96574e5fed33da74bc7855d5/src/cifar10/micro_child.py#L143
        x2 = x2[:, 1:, 1:, :]
        x2 = self.path2_avg_pool(x2, training=training)
        x2 = self.path2_conv(x2, training=training)

        x = tf.concat([x1, x2], axis=-1)
        return self.bn(x, training=training)


class CalibrateTwoLayersSize(tf.Module):
    
    def __init__(
            self, shape1: tuple, shape2: tuple, 
            out_filters: int, stride=2, **kwargs):
        super().__init__()
        stack_layers1 = []
        stack_layers2 = []
        
        if shape1[1] != shape2[1] or shape1[2] != shape2[2]:
            calibrate_layers = self._factorized_reduction(out_filters=out_filters, stride=stride, **kwargs)
            if shape1[1] * 2 == shape2[1] and shape1[2] * 2 == shape2[2]:
                stack_layers2 += calibrate_layers
            elif shape1[1] == shape2[1] * 2 and shape1[2] == shape2[2] * 2:
                stack_layers1 += calibrate_layers
        elif shape1[-1] != out_filters or shape2[-1] != out_filters:
            if shape1[-1] != out_filters:
                stack_layers1 += [
                    ConvBn(out_filters=out_filters, filter_size=(1,1), padding='SAME', **kwargs)
                ]
            if shape2[-1] != out_filters:
                stack_layers2 += [
                    ConvBn(out_filters=out_filters, filter_size=(1,1), padding='SAME', **kwargs)
                ]
        self.stacked_layers1_module, self.stacked_layers2_module = None, None
        if len(stack_layers1) > 0:
            self.stacked_layers1_module = StackedLayers(stack_layers1)
        if len(stack_layers2) > 0:
            self.stacked_layers2_module = StackedLayers(stack_layers2)
    
    @tf.function
    def __call__(self, x1, x2, training=False):
        if self.stacked_layers1_module:
            x1 = self.stacked_layers1_module(x1, training=training)
        if self.stacked_layers2_module:
            x2 = self.stacked_layers2_module(x2, training=training)
        return x1, x2

    def _factorized_reduction(self, out_filters, stride, **kwargs) -> list:
        assert out_filters % 2 == 0, f'out_filters={out_filters} must be divided by 2'
        if stride == 1:
            return [ConvBn(out_filters=out_filters, filter_size=(1,1), padding='SAME', **kwargs)]
        
        return [
            L.ReLU(),
            FactorizedReductionLayer(out_filters=out_filters, stride=stride, **kwargs),
        ]


class DropPathLayer(tf.Module):

    def __init__(self, layer_id: int, drop_path_keep_prob: float, num_layers: int, global_step: tf.Tensor):
        assert drop_path_keep_prob is not None and drop_path_keep_prob > 0 and drop_path_keep_prob < 1
        assert layer_id >= 0
        super().__init__()
        self.layer_id = layer_id
        self.drop_path_keep_prob = drop_path_keep_prob
        self.num_layers = num_layers
        self.global_step = global_step

    @tf.function
    def __call__(self, x, training=False):
        if not training:
            return x
        layer_ratio = tf.cast(self.layer_id + 1, dtype=tf.float32) / tf.cast(self.num_layers + 2, dtype=tf.float32)
        drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - self.drop_path_keep_prob)

        step_ratio = tf.cast(self.global_step + 1, dtype=tf.float32) / tf.cast(self.num_train_steps, dtype=tf.float32)
        step_ratio = tf.minimum(1.0, step_ratio)
        drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
        return apply_drop_path(x, drop_path_keep_prob)


class StackedLayers(tf.Module):

    def __init__(self, layers: list):
        super().__init__()
        self.layers = layers
    
    @tf.function
    def __call__(self, x, training=False):
        for layer in self.layers:
            x = layer(x, training=training)
        return x


class StackedRNN(tf.Module):

    def __init__(self, cells: list):
        assert len(cells) > 0
        super().__init__()
        self.cells = cells
    
    @tf.function
    def __call__(self, x, initial_state=None, training=False):
        for cell in self.cells:
            x, state_h, state_c = cell(
                x, initial_state=initial_state, 
                training=training
            )
            initial_state = (state_h, state_c)
        return x, state_h, state_c



class NASLayer(CheckpointRestoreModule):

    COUNT_OP = 5

    def __init__(
            self, layer_id: int, arc: Optional[list], out_filters: int, 
            num_cells: int, input_shape_list: List[tuple], global_step: tf.Tensor, 
            path_to_store_shared: str = None,
            mode=NASModelType.FIXED_MODE, drop_path_keep_prob: float=None,
            layer_reg=None, kernel_init='glorot_uniform',
            num_stack_enas_conv=2): 
        super().__init__()
        self.layer_id = layer_id
        self.arc = arc
        self.out_filters = out_filters
        self.num_cells = num_cells
        self.global_step = global_step
        self.path_to_store_shared = path_to_store_shared
        self.drop_path_keep_prob = drop_path_keep_prob
        self.num_stack_enas_conv = num_stack_enas_conv
        if arc is None and mode == NASModelType.FIXED_MODE:
            print('Mode for NASLyaer is Fixed, but arc is not provided. Swap mode to NAS-mode.')
            mode = NASModelType.NAS_MODE
        self.mode = mode

        self.calibrate_size_layer = None
        self.final_module = None
        self.base_module = None

        self.layer_reg = layer_reg
        self.kernel_init = kernel_init

        # List of lists of lists...
        # First index - layer_id
        self._enas_cells_created_list: List[list, List[list, tf.Module]] = [None] * num_cells
        for i in range(num_cells):
            self._enas_cells_created_list[i] = [  
                # Second index - branch id 
                [ # Third index - prev_cell (id)
                    [None] * NASLayer.COUNT_OP # Fourth index - op_id
                ] * (i + 2)
            ] * NASCellTypeV1.SIZE

        random_tensors = [
            tf.random.uniform(shape)
            for shape in input_shape_list
        ]
        self.is_builded = False
        self._build_layer(random_tensors)

    def load_by_parts(self, path_to_checkpoint: str):
        # Save separate modules:
        #   global_step, final_module, calibrate_size_layer, base_module
        try:
            tf.train.Checkpoint(
                global_step=self.global_step,
            ).read(path_to_checkpoint).expect_partial()
        except Exception as e:
            print(e)
            print('Skip global_step restore')

        try:
            tf.train.Checkpoint(
                final_module=self.final_module,
            ).read(path_to_checkpoint).expect_partial()
        except Exception as e:
            print(e)
            print('Skip final_module restore')

        try:
            tf.train.Checkpoint(
                calibrate_size_layer=self.calibrate_size_layer,
            ).read(path_to_checkpoint).expect_partial()
        except Exception as e:
            print(e)
            print('Skip calibrate_size_layer restore')

        if self.base_module is None and self.mode == NASModelType.FIXED_MODE:
            raise Exception('Base module is not builded but current mode is fixed.')

        if self.base_module is not None:
            try:
                tf.train.Checkpoint(
                    base_module=self.base_module,
                ).read(path_to_checkpoint).expect_partial()
            except Exception as e:
                print(e)
                print('Skip base_module restore')
        else:
            print('Skip base_module restore')
        # TODO: Load weights from shared folder
        # Load each layer independent from others. Why doing this?
        # While controller will be trained for our NASModel
        # different layers with configs will be produced, the main purpose of this lines
        # Is to load layers which are compatible with previous in order to stick to plan "Shared weights"
        # Shared aka these layers will be same, which means will be loaded from previous model
        for module_name, enas_cell in self._enas_cells_created_dict.items():
            try:
                if isinstance(enas_cell, StackedLayers):
                    for ii, layer in enumerate(enas_cell.layers):
                        try:
                            # Load path should be as:
                            #   '_enas_cells_created_dict/enas-cell-0-x-0-20/layers/1'
                            #   Here 'enas-cell-0-x-0-20' - is `module_name`
                            # This load try only for StackedLayers
                            tf.train.Checkpoint(
                                _enas_cells_created_dict={
                                    module_name: { 'layers': { str(ii): layer }}
                                }
                            ).read(path_to_checkpoint).expect_partial()
                        except Exception as e:
                            print(e)
                            print(f'Skip layer number equal to {ii} in {module_name} of stacked layers...')
                else:
                    # Load path should be as:
                    #   '_enas_cells_created_dict/enas-cell-0-x-0-20/layer'
                    #   Here 'enas-cell-0-x-0-20' - is `module_name`
                    tf.train.Checkpoint(
                        _enas_cells_created_dict={
                            module_name: enas_cell
                        }
                    ).read(path_to_checkpoint).expect_partial()
            except Exception as e:
                print(e)
                print('Skip module={module_name}')

    def _build_layer(self, prev_layers):
        assert len(prev_layers) == 2
        if self.is_builded:
            raise Exception('Layer is already builded!')
        # Answer will be resized two tensors from prev layers
        self.calibrate_size_layer = CalibrateTwoLayersSize(
            tf.shape(prev_layers[0]), tf.shape(prev_layers[1]), 
            self.out_filters,
            # Additional params for conv/bn
            kernel_regularizer=self.layer_reg,bias_regularizer=self.layer_reg,
            beta_regularizer=self.layer_reg,gamma_regularizer=self.layer_reg,
            kernel_initializer=self.kernel_init,
        )
        x0, x1 = self.calibrate_size_layer(prev_layers[0], prev_layers[1])
        prev_tensors = [x0, x1]

        if self.mode == NASModelType.FIXED_MODE:
            if self.base_module is None:
                self.base_module = StackedLayers([
                    ConvBn(
                        self.out_filters, (1,1), padding='SAME',
                        # Additional params for conv/bn
                        kernel_regularizer=self.layer_reg,bias_regularizer=self.layer_reg,
                        beta_regularizer=self.layer_reg,gamma_regularizer=self.layer_reg,
                        kernel_initializer=self.kernel_init,
                    )
                ])
            # TODO: `prev_tensors[1] =` but its seems like equal change  
            prev_tensors[-1] = self.base_module(prev_layers[-1])

        used = []
        for cell_id in range(self.num_cells):
            if self.arc is None and self.mode == NASModelType.NAS_MODE:
                x_id, x_op = 0, 0
            elif self.arc is not None and self.mode == NASModelType.FIXED_MODE:
                x_id = self.arc[4 * cell_id]
                x_op = self.arc[4 * cell_id + 1]
            else:
                # TODO: Give more details
                raise Exception('Unknown nas-layer X configuration.')
            x = tf.gather(prev_tensors, x_id) # prev_tensors[x_id]
            x = self._build_enas_cell(
                x, cell_id, x_id, x_op, self.out_filters, 
                cell_type=NASCellTypeV1.X_TYPE
            )

            if self.arc is None and self.mode == NASModelType.NAS_MODE:
                y_id, y_op = 0, 0
            elif self.arc is not None and self.mode == NASModelType.FIXED_MODE:
                y_id = self.arc[4 * cell_id + 2]
                y_op = self.arc[4 * cell_id + 3]
            else:
                # TODO: Give more details
                raise Exception('Unknown nas-layer Y configuration.')
            y = tf.gather(prev_tensors, y_id) # prev_tensors[y_id]
            y = self._build_enas_cell(
                y, cell_id, y_id, y_op, self.out_filters, 
                cell_type=NASCellTypeV1.Y_TYPE
            )

            merge_out = x + y
            prev_tensors.append(merge_out)

            used.extend([
                tf.one_hot(x_id, depth=self.num_cells + 2, dtype=tf.int32),
                tf.one_hot(y_id, depth=self.num_cells + 2, dtype=tf.int32)
            ])

        if self.mode == NASModelType.NAS_MODE:
            self.final_module = DynamicInputConvBn(
                self.num_cells+2, self.out_filters, 
                self.out_filters, (1,1), padding='SAME',
                # Additional params for conv/bn
                kernel_regularizer=self.layer_reg,bias_regularizer=self.layer_reg,
                beta_regularizer=self.layer_reg,gamma_regularizer=self.layer_reg,
                kernel_initializer=self.kernel_init,
            )

            # Select layers (outputs) which are not used for skip connection
            used = tf.add_n(used)
            indices = tf.where(tf.equal(used, 0))
            indices = tf.cast(indices, tf.int32)
            indices = tf.reshape(indices, [-1])
            num_outs = tf.size(indices)
            out = tf.stack(prev_tensors, axis=0)
            out = tf.gather(out, indices, axis=0)

            out = tf.transpose(out, [1, 2, 3, 0, 4])
            shape = tf.shape(out)
            out = tf.reshape(out, [shape[0], shape[1], shape[2], num_outs * self.out_filters])
        self.is_builded = True

    def _build_enas_conv(self, curr_cell, prev_cell, filter_size, out_filters, num_possible_inputs):
        builded_layers = []
        for conv_id in range(self.num_stack_enas_conv):
            builded_layers += [
                L.ReLU(),
                SeparableConvBn(
                    out_filters=out_filters, filter_size=filter_size, padding='SAME',
                    curr_cell=curr_cell, prev_cell=prev_cell, 
                    conv_id=conv_id, num_possible_inputs=num_possible_inputs,
                    # Additional params for conv/bn
                    kernel_regularizer=self.layer_reg,bias_regularizer=self.layer_reg,
                    beta_regularizer=self.layer_reg,gamma_regularizer=self.layer_reg,
                    kernel_initializer=self.kernel_init,
                )
            ]
        return builded_layers

    def _build_enas_cell_by_type(
            self, x, curr_cell, prev_cell, 
            op_id: int, out_filters: int, 
            num_possible_inputs: int):
        builded_layers = None
        if op_id == 0:
            builded_layers = self._build_enas_conv(
                curr_cell=curr_cell, prev_cell=prev_cell, 
                filter_size=5, out_filters=out_filters, 
                num_possible_inputs=num_possible_inputs
            )   
        elif op_id == 1:
            builded_layers = self._build_enas_conv(
                curr_cell=curr_cell, prev_cell=prev_cell, 
                filter_size=3, out_filters=out_filters, 
                num_possible_inputs=num_possible_inputs
            )
        elif op_id == 2:
            builded_layers = [
                L.AveragePooling2D((3,3), (1,1), padding='SAME')
            ]
            if tf.shape(x)[-1] != out_filters:
                builded_layers += [
                    ConvBn(
                        out_filters=out_filters, filter_size=(1,1), padding='SAME',
                        curr_cell=curr_cell, prev_cell=prev_cell, 
                        conv_id=0, num_possible_inputs=num_possible_inputs,
                        type_op=op_id,
                        # Additional params for conv/bn
                        kernel_regularizer=self.layer_reg,bias_regularizer=self.layer_reg,
                        beta_regularizer=self.layer_reg,gamma_regularizer=self.layer_reg,
                        kernel_initializer=self.kernel_init,
                    )
                ]
        elif op_id == 3:
            builded_layers = [
                L.MaxPool2D((3,3), (1,1), padding='SAME')
            ]
            if tf.shape(x)[-1] != out_filters:
                builded_layers += [
                    ConvBn(
                        out_filters=out_filters, filter_size=(1,1), padding='SAME',
                        curr_cell=curr_cell, prev_cell=prev_cell, 
                        conv_id=0, num_possible_inputs=num_possible_inputs,
                        type_op=op_id,
                        # Additional params for conv/bn
                        kernel_regularizer=self.layer_reg,bias_regularizer=self.layer_reg,
                        beta_regularizer=self.layer_reg,gamma_regularizer=self.layer_reg,
                        kernel_initializer=self.kernel_init,
                    )
                ]
        elif op_id == 4:
            builded_layers = []
            if tf.shape(x)[-1] != out_filters:
                builded_layers += [
                    ConvBn(
                        out_filters=out_filters, filter_size=(1,1), padding='SAME',
                        curr_cell=curr_cell, prev_cell=prev_cell, 
                        conv_id=0, num_possible_inputs=num_possible_inputs,
                        type_op=op_id,
                        # Additional params for conv/bn
                        kernel_regularizer=self.layer_reg,bias_regularizer=self.layer_reg,
                        beta_regularizer=self.layer_reg,gamma_regularizer=self.layer_reg,
                        kernel_initializer=self.kernel_init,
                    )
                ]
        if self.mode == NASModelType.FIXED_MODE and \
                op_id != 4 and self.drop_path_keep_prob is not None :
            
            builded_layers += [
                DropPathLayer(
                    self.layer_id, self.drop_path_keep_prob, 
                    self.num_layers, self.global_step
                )
            ]
        
        if builded_layers is None:
            raise Exception(f'In NAS-Layer `builded_layers` is None with op_id={op_id}')
        return StackedLayers(builded_layers)

    def _build_enas_cell(
            self, x, curr_cell, prev_cell, 
            op_id: int, out_filters: int, 
            cell_type: NASCellTypeV1, training=False):
        num_possible_inputs = curr_cell + 2
        if self.mode == NASModelType.NAS_MODE:
            """
            for prev_cell_i in range(num_possible_inputs):
                for op_id_s in range(NASLayer.COUNT_OP):
                    builded_layers_module_s = self._build_enas_cell_by_type(
                        x, curr_cell, prev_cell_i, op_id_s, out_filters, num_possible_inputs
                    )
                    # self._set_enas_cell(builded_layers_module_s, curr_cell, prev_cell, cell_type, op_id_s)
                    self._set_enas_cell(builded_layers_module_s, curr_cell, prev_cell_i, cell_type, op_id_s)
            """
            # TODO: Remove this. Only for test
            for op_id_s in range(NASLayer.COUNT_OP):
                builded_layers_module_s = self._build_enas_cell_by_type(
                    x, curr_cell, 0, op_id_s, out_filters, num_possible_inputs
                )
                # self._set_enas_cell(builded_layers_module_s, curr_cell, prev_cell, cell_type, op_id_s)
                self._set_enas_cell(builded_layers_module_s, curr_cell, 0, cell_type, op_id_s)
        elif self.mode == NASModelType.FIXED_MODE:
            builded_layers_module = self._build_enas_cell_by_type(
                x, curr_cell, prev_cell, op_id, out_filters, num_possible_inputs
            )
            self._set_enas_cell(builded_layers_module, curr_cell, prev_cell, cell_type, op_id)
        else:
            raise Exception('Unknown mode for nas-layer')
        return self.get_enas_cell(curr_cell, prev_cell, cell_type, op_id)(
            x, training=training
        )

    #@tf.function
    def __call__(self, prev_layers: List[tf.Tensor], arc=None, training=False):
        assert len(prev_layers) == 2 
        if arc is None and self.mode == NASModelType.NAS_MODE:
            raise Exception("Current layer mode is NAS-mode, but arc parameter was not provided to the layer.")
        elif arc is None and self.mode == NASModelType.FIXED_MODE:
            arc = self.arc
        elif arc is not None and self.mode == NASModelType.FIXED_MODE:
            raise Exception("Current layer mode is FIXED-mode, but arc parameter was provided to the layer.")
        # Answer will be resized two tensors from prev layers
        x0, x1 = self.calibrate_size_layer(
            *prev_layers, training=training
        )
        # TODO: Set size - its known here
        prev_tensors = [x0, x1]

        if self.mode == NASModelType.FIXED_MODE:
            if self.base_module is None:
                raise Exception('Base module is not builded but current mode is fixed.')
            prev_tensors[-1] = self.base_module(prev_layers[-1], training=training)

        used = []
        for cell_id in range(self.num_cells):
            x_id = arc[4 * cell_id]
            x_op = arc[4 * cell_id + 1]
            x = tf.gather(prev_tensors, x_id) # prev_tensors[x_id]
            x = self.get_enas_cell(
                cell_id, x_id, NASCellTypeV1.X_TYPE, x_op,
                x, training=training
            )


            y_id = arc[4 * cell_id + 2]
            y_op = arc[4 * cell_id + 3]
            y = tf.gather(prev_tensors, y_id) # prev_tensors[y_id]
            y = self.get_enas_cell(
                cell_id, y_id, NASCellTypeV1.Y_TYPE, y_op,
                x, training=training
            )

            merge_out = x + y
            prev_tensors.append(merge_out)

            used.extend([
                tf.one_hot(x_id, depth=self.num_cells + 2, dtype=tf.int32),
                tf.one_hot(y_id, depth=self.num_cells + 2, dtype=tf.int32)
            ])
        # Select layers (outputs) which are not used for skip connection
        used = tf.add_n(used)
        indices = tf.where(tf.equal(used, 0))
        indices = tf.cast(indices, tf.int32)
        indices = tf.reshape(indices, [-1])
        num_outs = tf.size(indices)
        out = tf.stack(prev_tensors, axis=0)
        out = tf.gather(out, indices, axis=0)

        out = tf.transpose(out, [1, 2, 3, 0, 4])
        shape = tf.shape(out)
        out = tf.reshape(out, [shape[0], shape[1], shape[2], num_outs * self.out_filters])

        if self.mode == NASModelType.NAS_MODE:
            if self.final_module is None:
                raise Exception('Final module is not builded but current mode is fixed.')
            out = self.final_module(out, indices=indices, training=training)
        return out
    
    def _set_enas_cell(self, module: tf.Module, curr_cell, prev_cell, cell_type, op_id):
        # TODO: Remove this. Only for test
        prev_cell = 0
        self._enas_cells_created_list[curr_cell][cell_type][prev_cell][op_id] = module

    def get_enas_cell(self, curr_cell, prev_cell, cell_type, op_id, x=None, training=None):
        # TODO: Call via:
        #       https://www.tensorflow.org/api_docs/python/tf/switch_case
        #       Should layers be stored as single big vector? Not like right now...
        try:
            # TODO: Remove this. Only for test
            prev_cell = 0
            if x is not None:
                return self._enas_cells_created_list[curr_cell][cell_type][prev_cell][op_id](
                    x, training=training
                )
            return self._enas_cells_created_list[curr_cell][cell_type][prev_cell][op_id]
        except Exception as e:
            print(f"Couldn't get enas-cell by curr_cell={curr_cell}, prev_cell={prev_cell}, cell_type={cell_type}, op_id={op_id}")
            raise e

    @tf.function
    def get_enas_cell_tf(
            self, curr_cell: int, 
            prev_cell: tf.Tensor, cell_type: int, 
            op_id: tf.Tensor, 
            *args, **kwargs):
        # TODO: Call via:
        #       https://www.tensorflow.org/api_docs/python/tf/switch_case
        #       Should layers be stored as single big vector? Not like right now...
        enas_cell_part_list = self._enas_cells_created_list[curr_cell][cell_type]
        layer_indx = NASLayer.COUNT_OP * prev_cell + op_id

        branch_fns_list = []
        # Num possible inputs equal to curr_cell + 2. Here `2` are inputs into this layer 
        for prev_cell_i in range(curr_cell + 2):
            for op_id_i in range(NASLayer.COUNT_OP):
                
                # Get enas cell call fn via function 
                # We could get it via lambda BUT value of variables `prev_cell_i` and `op_id_i`
                # Will be taken most-latest (aka at the end of this loop)
                # With this function - input variables are copied and its work as expected!
                def get_enas_call_fn(prev_cell_i, op_id_i):
                    return lambda: enas_cell_part_list[prev_cell_i][op_id_i](*args, **kwargs)

                branch_fns_list.append((
                    NASLayer.COUNT_OP * prev_cell_i + op_id_i, 
                    get_enas_call_fn(prev_cell_i, op_id_i),
                ))

        try:
            return tf.switch_case(layer_indx, branch_fns=branch_fns_list)
        except Exception as e:
            print(f"Couldn't get enas-cell by curr_cell={curr_cell}, prev_cell={prev_cell}, cell_type={cell_type}, op_id={op_id}")
            raise e


class AuxHeadLayer(CheckpointRestoreModule):

    def __init__(self, layer_id: int, proj_f=128, conv_f=768, n_classes=10):
        super().__init__()
        self.layer_id = layer_id

        self.proj = StackedLayers([
            L.ReLU(),
            L.AveragePooling2D((5,5), (3,3), 'VALID'),
            ConvBn(proj_f, (1,1), 'SAME', start_activation=None, end_activation=L.ReLU()),
        ])

        self.conv = ConvBn(conv_f, (1,1), 'SAME', start_activation=None, end_activation=L.ReLU())

        self.fc = StackedLayers([
            L.GlobalAveragePooling2D(),
            L.Dense(n_classes, use_bias=False),
        ])
    
    @tf.function
    def __call__(self, x, training=False):
        x = self.proj(x, training=training)
        x = self.conv(x, training=training)
        x = self.fc(x, training=training)
        return x

    def load_by_parts(self, path_to_checkpoint: str):
        # Save separate modules:
        #   proj, conv, fc
        try:
            tf.train.Checkpoint(
                proj=self.proj,
            ).read(path_to_checkpoint).expect_partial()
        except Exception as e:
            print(e)
            print(f'Skip proj restore in aug-head layer_id={self.layer_id}')

        try:
            tf.train.Checkpoint(
                conv=self.conv,
            ).read(path_to_checkpoint).expect_partial()
        except Exception as e:
            print(e)
            print(f'Skip conv restore in aug-head layer_id={self.layer_id}')

        try:
            tf.train.Checkpoint(
                fc=self.fc
            ).read(path_to_checkpoint).expect_partial()
        except Exception as e:
            print(e)
            print(f'Skip fc restore in aug-head layer_id={self.layer_id}')

