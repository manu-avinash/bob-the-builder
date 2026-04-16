import keras.layers as ly
import keras.activations as ac
class keras_Layers():
    def __init__(self):
        self.built={}
    #Base/core layers
    def dense(self,units=None,activation=None,use_bias=None,kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lora_rank=None,lora_alpha=None,quantization_config=None,**kwargs):
        layer=ly.Dense(units=units,activation=activation,use_bias=use_bias,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint,lora_rank=lora_rank,lora_alpha=lora_alpha,quantization_config=quantization_config,**kwargs)
        self.built[layer.name]=layer
        return layer
    def input_(self,shape=None,batch_size=None,dtype=None,sparse=None,ragged=None,batch_shape=None,name=None,tensor=None,optional=None):
        layer= ly.Input(shape=shape,batch_size=batch_size,dtype=dtype,sparse=sparse,ragged=ragged,batch_shape=batch_shape,name=name,tensor=tensor,optional=optional)
        self.built[layer.name]=layer
        return layer
    def input_spec(self,dtype=None,shape=None,ndim=None,max_ndim=None,min_ndim=None,axes=None,allow_last_axis_squeeze=None,name=None,optional=None):
        layer= ly.InputSpec(dtype=dtype,shape=shape,ndim=ndim,max_ndim=max_ndim,min_ndim=min_ndim,axes=axes,allow_last_axis_squeeze=allow_last_axis_squeeze,name=name,optional=optional)
        self.built[layer.name]=layer
        return layer
    def einsumdense(self,equation=None,output_shape=None,activation=None,bias_axes=None,kernel_initializer=None,bias_initializer=None,kernel_regularizer=None,bias_regularizer=None,kernel_constraint=None,bias_constraint=None,lora_rank=None,lora_alpha=None,gptq_unpacked_column_size=None,quantization_config=None,**kwargs):
        layer= ly.EinsumDense(equation=equation,output_shape=output_shape,activation=activation,bias_axes=bias_axes,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint,lora_rank=lora_rank,lora_alpha=lora_alpha,gptq_unpacked_column_size=gptq_unpacked_column_size,quantization_config=quantization_config,**kwargs)
        self.built[layer.name]=layer
        return layer
    def embedding(self,input_dim=None,output_dim=None,embeddings_initializer=None,embeddings_regularizer=None,embeddings_constraint=None,mask_zero=None,weights=None,lora_rank=None,lora_alpha=None,quantization_config=None,**kwargs):
        layer= ly.Embedding(input_dim=input_dim,output_dim=output_dim,embeddings_initializer=embeddings_initializer,embeddings_regularizer=embeddings_regularizer,embeddings_constraint=embeddings_constraint,mask_zero=mask_zero,weights=weights,lora_rank=lora_rank,lora_alpha=lora_alpha,quantization_config=quantization_config,**kwargs)
        self.built[layer.name]=layer
        return layer
    def reversibleEmbedding(self,input_dim=None,output_dim=None,tie_weights=None,embeddings_initializer=None,embeddings_regularizer=None,embeddings_constraint=None,mask_zero=None,reverse_dtype=None,logit_soft_cap=None,**kwargs):
        layer= ly.ReversibleEmbedding(input_dim=input_dim,output_dim=output_dim,tie_weights=tie_weights,embeddings_initializer=embeddings_initializer,embeddings_regularizer=embeddings_regularizer,embeddings_constraint=embeddings_constraint,mask_zero=mask_zero,reverse_dtype=reverse_dtype,logit_soft_cap=logit_soft_cap,**kwargs)
        self.built[layer.name]=layer
        return layer
    def masking(self,mask_value=None,**kwargs):
        layer= ly.Masking(mask_value=mask_value,**kwargs)
        self.built[layer.name]=layer
        return layer
    def lambda_(self,function=None, output_shape=None, mask=None, arguments=None, **kwargs):
        layer= ly.Lambda(function=function, output_shape=output_shape, mask=mask, arguments=arguments, **kwargs)
        self.built[layer.name]=layer
        return layer
    def identity(self,**kwargs):
        layer= ly.Identity(**kwargs)
        self.built[layer.name]=layer
        return layer
    #Image and graphics - convolution
    def conv1d(self,filters=None,kernel_size=None,strides=None,padding=None,data_format=None,dilation_rate=None,groups=None,activation=None,use_bias=None,kernel_initializer=None,bias_initializer=None,kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,**kwargs):
        layer=ly.Conv1D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,groups=groups,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint,**kwargs)
        self.built[layer.name]=layer
        return layer
    def conv2d(self,filters=None,kernel_size=None,strides=None,padding=None,data_format=None,dilation_rate=None,groups=None,activation=None,use_bias=None,kernel_initializer=None,bias_initializer=None,kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,**kwargs):
        layer=ly.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,groups=groups,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint,**kwargs)
        self.built[layer.name]=layer
        return layer
    def conv3d(self,filters=None,kernel_size=None,strides=None,padding=None,data_format=None,dilation_rate=None,groups=None,activation=None,use_bias=None,kernel_initializer=None,bias_initializer=None,kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,**kwargs):
        layer=ly.Conv3D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,groups=groups,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint,**kwargs)
        self.built[layer.name]=layer
        return layer
    def separable_conv1d(self,filters=None,kernel_size=None,strides=None,padding=None,data_format=None,dilation_rate=None,depth_multiplier=None,activation=None,use_bias=None,depthwise_initializer=None,pointwise_initializer=None,bias_initializer=None,depthwise_regularizer=None,pointwise_regularizer=None,bias_regularizer=None,activity_regularizer=None,depthwise_constraint=None,pointwise_constraint=None,bias_constraint=None,**kwargs):
        layer=ly.SeparableConv1D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,depth_multiplier=depth_multiplier,activation=activation,use_bias=use_bias,depthwise_initializer=depthwise_initializer,pointwise_initializer=pointwise_initializer,bias_initializer=bias_initializer,depthwise_regularizer=depthwise_regularizer,pointwise_regularizer=pointwise_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,depthwise_constraint=depthwise_constraint,pointwise_constraint=pointwise_constraint,bias_constraint=bias_constraint,**kwargs)
        self.built[layer.name]=layer
        return layer
    def separable_conv2d(self,filters=None,kernel_size=None,strides=None,padding=None,data_format=None,dilation_rate=None,depth_multiplier=None,activation=None,use_bias=None,depthwise_initializer=None,pointwise_initializer=None,bias_initializer=None,depthwise_regularizer=None,pointwise_regularizer=None,bias_regularizer=None,activity_regularizer=None,depthwise_constraint=None,pointwise_constraint=None,bias_constraint=None,**kwargs):
        layer=ly.SeparableConv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,depth_multiplier=depth_multiplier,activation=activation,use_bias=use_bias,depthwise_initializer=depthwise_initializer,pointwise_initializer=pointwise_initializer,bias_initializer=bias_initializer,depthwise_regularizer=depthwise_regularizer,pointwise_regularizer=pointwise_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,depthwise_constraint=depthwise_constraint,pointwise_constraint=pointwise_constraint,bias_constraint=bias_constraint,**kwargs)
        self.built[layer.name]=layer
        return layer
    def depthwiseconv1d(self,kernel_size=None,strides=None,padding=None,depth_multiplier=None,data_format=None,dilation_rate=None,activation=None,use_bias=None,depthwise_initializer=None,bias_initializer=None,depthwise_regularizer=None,bias_regularizer=None,activity_regularizer=None,depthwise_constraint=None,bias_constraint=None,**kwargs):
        layer=ly.DepthwiseConv1D(kernel_size=kernel_size,strides=strides,padding=padding,depth_multiplier=depth_multiplier,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,depthwise_initializer=depthwise_initializer,bias_initializer=bias_initializer,depthwise_regularizer=depthwise_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,depthwise_constraint=depthwise_constraint,bias_constraint=bias_constraint,**kwargs)
        self.built[layer.name]=layer
        return layer
    def depthwiseconv2d(self,kernel_size=None,strides=None,padding=None,depth_multiplier=None,data_format=None,dilation_rate=None,activation=None,use_bias=None,depthwise_initializer=None,bias_initializer=None,depthwise_regularizer=None,bias_regularizer=None,activity_regularizer=None,depthwise_constraint=None,bias_constraint=None,**kwargs):
        layer=ly.DepthwiseConv2D(kernel_size=kernel_size,strides=strides,padding=padding,depth_multiplier=depth_multiplier,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,depthwise_initializer=depthwise_initializer,bias_initializer=bias_initializer,depthwise_regularizer=depthwise_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,depthwise_constraint=depthwise_constraint,bias_constraint=bias_constraint,**kwargs)
        self.built[layer.name]=layer
        return layer
    def conv1d_transpose(self,filters=None,kernel_size=None,strides=None,padding=None,output_padding=None,data_format=None,dilation_rate=None,activation=None,use_bias=None,kernel_initializer=None,bias_initializer=None,kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,**kwargs):
        layer=ly.Conv1DTranspose(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,output_padding=output_padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint,**kwargs)
        self.built[layer.name]=layer
        return layer
    def conv2d_transpose(self,filters=None,kernel_size=None,strides=None,padding=None,output_padding=None,data_format=None,dilation_rate=None,activation=None,use_bias=None,kernel_initializer=None,bias_initializer=None,kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,**kwargs):
        layer=ly.Conv2DTranspose(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,output_padding=output_padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint,**kwargs)
        self.built[layer.name]=layer
        return layer
    def conv3d_transpose(self,filters=None,kernel_size=None,strides=None,padding=None,data_format=None,output_padding=None,dilation_rate=None,activation=None,use_bias=None,kernel_initializer=None,bias_initializer=None,kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,**kwargs):
        layer=ly.Conv3DTranspose(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,output_padding=output_padding,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint,**kwargs)
        self.built[layer.name]=layer
        return layer
    def maxpool1d(self,pool_size=None, strides=None, padding=None, data_format=None, name=None, **kwargs):
        layer=ly.MaxPool1D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, name=name, **kwargs)
        self.built[layer.name]=layer
        return layer
    def maxpool2d(self,pool_size=None, strides=None, padding=None, data_format=None, name=None, **kwargs):
        layer=ly.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, name=name, **kwargs)
        self.built[layer.name]=layer
        return layer
    def maxpool3d(self,pool_size=None,strides=None,padding=None,data_format=None,name=None,**kwargs):
        layer=ly.MaxPooling3D(pool_size=pool_size,strides=strides,padding=padding,data_format=data_format,name=name,**kwargs)
        self.built[layer.name]=layer
        return layer
    def averagepooling1d(self,pool_size=None, strides=None, padding=None, data_format=None, name=None, **kwargs):
        layer=ly.AveragePooling1D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, name=name, **kwargs)
        self.built[layer.name]=layer
        return layer
    def averagepooling2d(self,pool_size=None, strides=None, padding=None, data_format=None, name=None, **kwargs):
        layer=ly.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, name=name, **kwargs)
        self.built[layer.name]=layer
        return layer
    def averagepooling3d(self,pool_size=None, strides=None, padding=None, data_format=None, name=None, **kwargs):
        layer=ly.AveragePooling3D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, name=name, **kwargs)
        self.built[layer.name]=layer
        return layer
    def globalmaxpooling1d(self,data_format=None, keepdims=None, **kwargs):
        layer=ly.GlobalMaxPooling1D(data_format=data_format, keepdims=keepdims, **kwargs)
        self.built[layer.name]=layer
        return layer
    def globalmaxpooling2d(self,data_format=None, keepdims=None, **kwargs):
        layer=ly.GlobalMaxPooling2D(data_format=data_format, keepdims=keepdims, **kwargs)
        self.built[layer.name]=layer
        return layer
    def globalmaxpooling3d(self,data_format=None, keepdims=None, **kwargs):
        layer=ly.GlobalMaxPooling3D(data_format=data_format, keepdims=keepdims, **kwargs)
        self.built[layer.name]=layer
        return layer
    def globalaveragepooling1d(self,data_format=None, keepdims=None, **kwargs):
        layer=ly.GlobalAveragePooling1D(data_format=data_format, keepdims=keepdims, **kwargs)
        self.built[layer.name]=layer
        return layer
    def globalaveragepooling2d(self,data_format=None, keepdims=None, **kwargs):
        layer=ly.GlobalAveragePooling2D(data_format=data_format, keepdims=keepdims, **kwargs)
        self.built[layer.name]=layer
        return layer
    def globalaveragepooling3d(self,data_format=None, keepdims=None, **kwargs):
        layer=ly.GlobalAveragePooling3D(data_format=data_format, keepdims=keepdims, **kwargs)
        self.built[layer.name]=layer
        return layer
    def adaptiveaveragepooling1d(self,output_size=None, data_format=None, **kwargs):
        layer=ly.AdaptiveAveragePooling1D(output_size=output_size, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def adaptiveaveragepooling2d(self,output_size=None, data_format=None, **kwargs):
        layer=ly.AdaptiveAveragePooling2D(output_size=output_size, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def adaptiveaveragepooling3d(self,output_size=None, data_format=None, **kwargs):
        layer=ly.AdaptiveAveragePooling3D(output_size=output_size, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def adaptivemaxpooling1d(self,output_size=None, data_format=None, **kwargs):
        layer=ly.AdaptiveMaxPooling1D(output_size=output_size, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def adaptivemaxpooling2d(self,output_size=None, data_format=None, **kwargs):
        layer=ly.AdaptiveMaxPooling2D(output_size=output_size, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def adaptivemaxpooling3d(self,output_size=None, data_format=None, **kwargs):
        layer=ly.AdaptiveMaxPooling3D(output_size=output_size, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    #nlp-lstm
    def lstm(self,units=None,activation=None,recurrent_activation=None,use_bias=None,kernel_initializer=None,recurrent_initializer=None,bias_initializer=None,unit_forget_bias=None,kernel_regularizer=None,recurrent_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,recurrent_constraint=None,bias_constraint=None,dropout=None,recurrent_dropout=None,seed=None,return_sequences=None,return_state=None,go_backwards=None,stateful=None,unroll=None,use_cudnn=None,**kwargs):
        layer=ly.LSTM(units=units,activation=activation,recurrent_activation=recurrent_activation,use_bias=use_bias,kernel_initializer=kernel_initializer,recurrent_initializer=recurrent_initializer,bias_initializer=bias_initializer,unit_forget_bias=unit_forget_bias,kernel_regularizer=kernel_regularizer,recurrent_regularizer=recurrent_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,recurrent_constraint=recurrent_constraint,bias_constraint=bias_constraint,dropout=dropout,recurrent_dropout=recurrent_dropout,seed=seed,return_sequences=return_sequences,return_state=return_state,go_backwards=go_backwards,stateful=stateful,unroll=unroll,use_cudnn=use_cudnn,**kwargs)
        self.built[layer.name]=layer
        return layer
    def lstm_cell(self,units=None,activation=None,recurrent_activation=None,use_bias=None,kernel_initializer=None,recurrent_initializer=None,bias_initializer=None,unit_forget_bias=None,kernel_regularizer=None,recurrent_regularizer=None,bias_regularizer=None,kernel_constraint=None,recurrent_constraint=None,bias_constraint=None,dropout=None,recurrent_dropout=None,seed=None,**kwargs):
        layer=ly.LSTMCell(units=units,activation=activation,recurrent_activation=recurrent_activation,use_bias=use_bias,kernel_initializer=kernel_initializer,recurrent_initializer=recurrent_initializer,bias_initializer=bias_initializer,unit_forget_bias=unit_forget_bias,kernel_regularizer=kernel_regularizer,recurrent_regularizer=recurrent_regularizer,bias_regularizer=bias_regularizer,kernel_constraint=kernel_constraint,recurrent_constraint=recurrent_constraint,bias_constraint=bias_constraint,dropout=dropout,recurrent_dropout=recurrent_dropout,seed=seed,**kwargs)
        self.built[layer.name]=layer
        return layer
    def gru(self,units=None,activation=None,recurrent_activation=None,use_bias=None,kernel_initializer=None,recurrent_initializer=None,bias_initializer=None,kernel_regularizer=None,recurrent_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,recurrent_constraint=None,bias_constraint=None,dropout=None,recurrent_dropout=None,seed=None,return_sequences=None,return_state=None,go_backwards=None,stateful=None,unroll=None,reset_after=None,use_cudnn=None,**kwargs):
        layer=ly.GRU(units=units,activation=activation,recurrent_activation=recurrent_activation,use_bias=use_bias,kernel_initializer=kernel_initializer,recurrent_initializer=recurrent_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,recurrent_regularizer=recurrent_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,recurrent_constraint=recurrent_constraint,bias_constraint=bias_constraint,dropout=dropout,recurrent_dropout=recurrent_dropout,seed=seed,return_sequences=return_sequences,return_state=return_state,go_backwards=go_backwards,stateful=stateful,unroll=unroll,reset_after=reset_after,use_cudnn=use_cudnn,**kwargs)
        self.built[layer.name]=layer
        return layer
    def gru_cell(self,units=None,activation=None,recurrent_activation=None,use_bias=None,kernel_initializer=None,recurrent_initializer=None,bias_initializer=None,kernel_regularizer=None,recurrent_regularizer=None,bias_regularizer=None,kernel_constraint=None,recurrent_constraint=None,bias_constraint=None,dropout=None,recurrent_dropout=None,reset_after=None,seed=None,**kwargs):
        layer=ly.GRUCell(units=units,activation=activation,recurrent_activation=recurrent_activation,use_bias=use_bias,kernel_initializer=kernel_initializer,recurrent_initializer=recurrent_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,recurrent_regularizer=recurrent_regularizer,bias_regularizer=bias_regularizer,kernel_constraint=kernel_constraint,recurrent_constraint=recurrent_constraint,bias_constraint=bias_constraint,dropout=dropout,recurrent_dropout=recurrent_dropout,reset_after=reset_after,seed=seed,**kwargs)
        self.built[layer.name]=layer
        return layer
    def simpleRnn(self,units=None,activation=None,use_bias=None,kernel_initializer=None,recurrent_initializer=None,bias_initializer=None,kernel_regularizer=None,recurrent_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,recurrent_constraint=None,bias_constraint=None,dropout=None,recurrent_dropout=None,return_sequences=None,return_state=None,go_backwards=None,stateful=None,unroll=None,seed=None,**kwargs):
        layer=ly.SimpleRNN(units=units,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,recurrent_initializer=recurrent_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,recurrent_regularizer=recurrent_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,recurrent_constraint=recurrent_constraint,bias_constraint=bias_constraint,dropout=dropout,recurrent_dropout=recurrent_dropout,return_sequences=return_sequences,return_state=return_state,go_backwards=go_backwards,stateful=stateful,unroll=unroll,seed=seed,**kwargs)
        self.built[layer.name]=layer
        return layer
    def timedistributed(self,layer=None,**kwargs):
        layer=ly.TimeDistributed(layer=layer,**kwargs)
        self.built[layer.name]=layer
        return layer
    def bidirectional(self,layer=None, merge_mode=None, weights=None, backward_layer=None, **kwargs):
        layer=ly.Bidirectional(layer=layer, merge_mode=merge_mode, weights=weights, backward_layer=backward_layer, **kwargs)
        self.built[layer.name]=layer
        return layer
    #image and graphics-lstm
    def convlstm1d(self,filters=None,kernel_size=None,strides=None,padding=None,data_format=None,dilation_rate=None,activation=None,recurrent_activation=None,use_bias=None,kernel_initializer=None,recurrent_initializer=None,bias_initializer=None,unit_forget_bias=None,kernel_regularizer=None,recurrent_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,recurrent_constraint=None,bias_constraint=None,dropout=None,recurrent_dropout=None,seed=None,return_sequences=None,return_state=None,go_backwards=None,stateful=None,**kwargs):
        layer=ly.ConvLSTM1D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,recurrent_activation=recurrent_activation,use_bias=use_bias,kernel_initializer=kernel_initializer,recurrent_initializer=recurrent_initializer,bias_initializer=bias_initializer,unit_forget_bias=unit_forget_bias,kernel_regularizer=kernel_regularizer,recurrent_regularizer=recurrent_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,recurrent_constraint=recurrent_constraint,bias_constraint=bias_constraint,dropout=dropout,recurrent_dropout=recurrent_dropout,seed=seed,return_sequences=return_sequences,return_state=return_state,go_backwards=go_backwards,stateful=stateful,**kwargs)
        self.built[layer.name]=layer
        return layer
    def convlstm2d(self,filters=None,kernel_size=None,strides=None,padding=None,data_format=None,dilation_rate=None,activation=None,recurrent_activation=None,use_bias=None,kernel_initializer=None,recurrent_initializer=None,bias_initializer=None,unit_forget_bias=None,kernel_regularizer=None,recurrent_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,recurrent_constraint=None,bias_constraint=None,dropout=None,recurrent_dropout=None,seed=None,return_sequences=None,return_state=None,go_backwards=None,stateful=None,**kwargs):
        layer=ly.ConvLSTM2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,recurrent_activation=recurrent_activation,use_bias=use_bias,kernel_initializer=kernel_initializer,recurrent_initializer=recurrent_initializer,bias_initializer=bias_initializer,unit_forget_bias=unit_forget_bias,kernel_regularizer=kernel_regularizer,recurrent_regularizer=recurrent_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,recurrent_constraint=recurrent_constraint,bias_constraint=bias_constraint,dropout=dropout,recurrent_dropout=recurrent_dropout,seed=seed,return_sequences=return_sequences,return_state=return_state,go_backwards=go_backwards,stateful=stateful,**kwargs)
        self.built[layer.name]=layer
        return layer
    def convlstm3d(self,filters=None,kernel_size=None,strides=None,padding=None,data_format=None,dilation_rate=None,activation=None,recurrent_activation=None,use_bias=None,kernel_initializer=None,recurrent_initializer=None,bias_initializer=None,unit_forget_bias=None,kernel_regularizer=None,recurrent_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,recurrent_constraint=None,bias_constraint=None,dropout=None,recurrent_dropout=None,seed=None,return_sequences=None,return_state=None,go_backwards=None,stateful=None,**kwargs):
        layer=ly.ConvLSTM3D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,recurrent_activation=recurrent_activation,use_bias=use_bias,kernel_initializer=kernel_initializer,recurrent_initializer=recurrent_initializer,bias_initializer=bias_initializer,unit_forget_bias=unit_forget_bias,kernel_regularizer=kernel_regularizer,recurrent_regularizer=recurrent_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,recurrent_constraint=recurrent_constraint,bias_constraint=bias_constraint,dropout=dropout,recurrent_dropout=recurrent_dropout,seed=seed,return_sequences=return_sequences,return_state=return_state,go_backwards=go_backwards,stateful=stateful,**kwargs)
        self.built[layer.name]=layer
        return layer
    #nlp-Rnn
    def baseRnn(self,cell=None,return_sequences=None,return_state=None,go_backwards=None,stateful=None,unroll=None,zero_output_for_mask=None,**kwargs):
        layer=ly.RNN(cell=cell,return_sequences=return_sequences,return_state=return_state,go_backwards=go_backwards,stateful=stateful,unroll=unroll,zero_output_for_mask=zero_output_for_mask,**kwargs)
        self.built[layer.name]=layer
        return layer
    def simpleRnnCell(self,units=None,activation=None,use_bias=None,kernel_initializer=None,recurrent_initializer=None,bias_initializer=None,kernel_regularizer=None,recurrent_regularizer=None,bias_regularizer=None,kernel_constraint=None,recurrent_constraint=None,bias_constraint=None,dropout=None,recurrent_dropout=None,seed=None,**kwargs):
        layer=ly.SimpleRNNCell(units=units,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,recurrent_initializer=recurrent_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,recurrent_regularizer=recurrent_regularizer,bias_regularizer=bias_regularizer,kernel_constraint=kernel_constraint,recurrent_constraint=recurrent_constraint,bias_constraint=bias_constraint,dropout=dropout,recurrent_dropout=recurrent_dropout,seed=seed,**kwargs)
        self.built[layer.name]=layer
        return layer
    def stackedRnnCell(self,cells=None, **kwargs):
        layer=ly.StackedRNNCells(cells=cells, **kwargs)
        self.built[layer.name]=layer
        return layer
    #preprocessing-text
    def textvectorization(self,max_tokens=None,standardize=None,split=None,ngrams=None,output_mode=None,output_sequence_length=None,pad_to_max_tokens=None,vocabulary=None,idf_weights=None,sparse=None,ragged=None,encoding=None,name=None,**kwargs):
        layer=ly.TextVectorization(max_tokens=max_tokens,standardize=standardize,split=split,ngrams=ngrams,output_mode=output_mode,output_sequence_length=output_sequence_length,pad_to_max_tokens=pad_to_max_tokens,vocabulary=vocabulary,idf_weights=idf_weights,sparse=sparse,ragged=ragged,encoding=encoding,name=name,**kwargs)
        self.built[layer.name]=layer
        return layer
    #preprocessing-numerical feature 
    def normalization(self,axis=None, mean=None, variance=None, invert=None, **kwargs):
        layer=ly.Normalization(axis=axis, mean=mean, variance=variance, invert=invert, **kwargs)
        self.built[layer.name]=layer
        return layer
    def spectral_normalization(self,layer=None, power_iterations=None, **kwargs):
        layer=ly.SpectralNormalization(layer=layer, power_iterations=power_iterations, **kwargs)
        self.built[layer.name]=layer
        return layer
    def discretization(self,bin_boundaries=None,num_bins=None,epsilon=None,output_mode=None,sparse=None,dtype=None,name=None):
        layer=ly.Discretization(bin_boundaries=bin_boundaries,num_bins=num_bins,epsilon=epsilon,output_mode=output_mode,sparse=sparse,dtype=dtype,name=name)
        self.built[layer.name]=layer
        return layer
    #preprocessing-Categorical features
    def category_encoding(self,num_tokens=None, output_mode=None, sparse=None, **kwargs):
        layer=ly.CategoryEncoding(num_tokens=num_tokens, output_mode=output_mode, sparse=sparse, **kwargs)
        self.built[layer.name]=layer
        return layer
    def hashing(self,num_bins=None, mask_value=None, salt=None, output_mode=None, sparse=None, **kwargs):
        layer=ly.Hashing(num_bins=num_bins, mask_value=mask_value, salt=salt, output_mode=output_mode, sparse=sparse, **kwargs)
        self.built[layer.name]=layer
        return layer
    def hashed_crossing(self,num_bins=None, output_mode=None, sparse=None, name=None, dtype=None, **kwargs):
        layer=ly.HashedCrossing(num_bins=num_bins, output_mode=output_mode, sparse=sparse, name=name, dtype=dtype, **kwargs)
        self.built[layer.name]=layer
        return layer
    def stringlookup(self,max_tokens=None,num_oov_indices=None,mask_token=None,oov_token=None,vocabulary=None,idf_weights=None,invert=None,output_mode=None,pad_to_max_tokens=None,sparse=None,encoding=None,name=None,salt=None,**kwargs):
        layer=ly.StringLookup(max_tokens=max_tokens,num_oov_indices=num_oov_indices,mask_token=mask_token,oov_token=oov_token,vocabulary=vocabulary,idf_weights=idf_weights,invert=invert,output_mode=output_mode,pad_to_max_tokens=pad_to_max_tokens,sparse=sparse,encoding=encoding,name=name,salt=salt,**kwargs)
        self.built[layer.name]=layer
        return layer
    def integerlookup(self,max_tokens=None,num_oov_indices=None,mask_token=None,oov_token=None,vocabulary=None,vocabulary_dtype=None,idf_weights=None,invert=None,output_mode=None,sparse=None,pad_to_max_tokens=None,oov_method=None,salt=None,name=None,**kwargs):
        layer=ly.IntegerLookup(max_tokens=max_tokens,num_oov_indices=num_oov_indices,mask_token=mask_token,oov_token=oov_token,vocabulary=vocabulary,vocabulary_dtype=vocabulary_dtype,idf_weights=idf_weights,invert=invert,output_mode=output_mode,sparse=sparse,pad_to_max_tokens=pad_to_max_tokens,oov_method=oov_method,salt=salt,name=name,**kwargs)
        self.built[layer.name]=layer
        return layer
    #preprocessing-image
    def resizing(self,height=None,width=None,interpolation=None,crop_to_aspect_ratio=None,pad_to_aspect_ratio=None,fill_mode=None,fill_value=None,antialias=None,data_format=None,**kwargs):
        layer=ly.Resizing(height=height,width=width,interpolation=interpolation,crop_to_aspect_ratio=crop_to_aspect_ratio,pad_to_aspect_ratio=pad_to_aspect_ratio,fill_mode=fill_mode,fill_value=fill_value,antialias=antialias,data_format=data_format,**kwargs)
        self.built[layer.name]=layer
        return layer
    def rescaling(self,scale=None, offset=None, **kwargs):
        layer=ly.Rescaling(scale=scale, offset=offset, **kwargs)
        self.built[layer.name]=layer
        return layer
    def centercrop(self,height=None, width=None, data_format=None, **kwargs):
        layer=ly.CenterCrop(height=height, width=width, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def autocontrast(self,value_range=None, **kwargs):
        layer=ly.AutoContrast(value_range=value_range, **kwargs)
        self.built[layer.name]=layer
        return layer
    #preprocessing-image augmentation
    def augmix(self,value_range=None,num_chains=None,chain_depth=None,factor=None,alpha=None,all_ops=None,interpolation=None,seed=None,data_format=None,**kwargs):
        layer=ly.AugMix(value_range=value_range,num_chains=num_chains,chain_depth=chain_depth,factor=factor,alpha=alpha,all_ops=all_ops,interpolation=interpolation,seed=seed,data_format=data_format,**kwargs)
        self.built[layer.name]=layer
        return layer
    def cutmix(self,factor=None, seed=None, data_format=None, **kwargs):
        layer=ly.CutMix(factor=factor, seed=seed, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def equalization(self,value_range=None, bins=None, data_format=None, **kwargs):
        layer=ly.Equalization(value_range=value_range, bins=bins, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def maxnumboundingboxes(self,max_number=None, fill_value=None, **kwargs):
        layer=ly.MaxNumBoundingBoxes(max_number=max_number, fill_value=fill_value, **kwargs)
        self.built[layer.name]=layer
        return layer
    def mixUp(self,alpha=None, data_format=None, seed=None, **kwargs):
        layer=ly.MixUp(alpha=alpha, data_format=data_format, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    def pipeline(self,layers=None, name=None):
        layer=ly.Pipeline(layers=layers, name=name)
        self.built[layer.name]=layer
        return layer
    def randAugment(self,value_range=None,num_ops=None,factor=None,interpolation=None,seed=None,data_format=None,**kwargs):
        layer=ly.RandAugment(value_range=value_range,num_ops=num_ops,factor=factor,interpolation=interpolation,seed=seed,data_format=data_format,**kwargs)
        self.built[layer.name]=layer
        return layer
    def randomBrightness(self,factor=None, value_range=None, seed=None, **kwargs):
        layer=ly.RandomBrightness(factor=factor, value_range=value_range, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    def randomColorDegeneration(self,factor=None, value_range=None, data_format=None, seed=None, **kwargs):
        layer=ly.RandomColorDegeneration(factor=factor, value_range=value_range, data_format=data_format, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    def randomColorJitter(self,value_range=None,brightness_factor=None,contrast_factor=None,saturation_factor=None,hue_factor=None,seed=None,data_format=None,**kwargs):
        layer=ly.RandomColorJitter(value_range=value_range,brightness_factor=brightness_factor,contrast_factor=contrast_factor,saturation_factor=saturation_factor,hue_factor=hue_factor,seed=seed,data_format=data_format,**kwargs)
        self.built[layer.name]=layer
        return layer
    def randomContrast(self,factor=None, value_range=None, seed=None, **kwargs):
        layer=ly.RandomContrast(factor=factor, value_range=value_range, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    def randomCrop(self,height=None, width=None, seed=None, data_format=None, name=None, **kwargs):
        layer=ly.RandomCrop(height=height, width=width, seed=seed, data_format=data_format, name=name, **kwargs)
        self.built[layer.name]=layer
        return layer
    def randomElasticTransform(self,factor=None,scale=None,interpolation=None,fill_mode=None,fill_value=None,value_range=None,seed=None,data_format=None,**kwargs):
        layer=ly.RandomElasticTransform(factor=factor,scale=scale,interpolation=interpolation,fill_mode=fill_mode,fill_value=fill_value,value_range=value_range,seed=seed,data_format=data_format,**kwargs)
        self.built[layer.name]=layer
        return layer
    def randomErasing(self,factor=None,scale=None,fill_value=None,value_range=None,seed=None,data_format=None,**kwargs):
        layer=ly.RandomErasing(factor=factor,scale=scale,fill_value=fill_value,value_range=value_range,seed=seed,data_format=data_format,**kwargs)
        self.built[layer.name]=layer
        return layer
    def randomFlip(self,mode=None, seed=None, data_format=None, **kwargs):
        layer=ly.RandomFlip(mode=mode, seed=seed, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def randomGaussianBlur(self,factor=None,kernel_size=None,sigma=None,value_range=None,data_format=None,seed=None,**kwargs):
        layer=ly.RandomGaussianBlur(factor=factor,kernel_size=kernel_size,sigma=sigma,value_range=value_range,data_format=data_format,seed=seed,**kwargs)
        self.built[layer.name]=layer
        return layer
    def randomGrayscale(self,factor=None, data_format=None, seed=None, **kwargs):
        layer=ly.RandomGrayscale(factor=factor, data_format=data_format, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    def randomHue(self,factor=None, value_range=None, data_format=None, seed=None, **kwargs):
        layer=ly.RandomHue(factor=factor, value_range=value_range, data_format=data_format, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    def randomInvert(self,factor=None, value_range=None, seed=None, data_format=None, **kwargs):
        layer=ly.RandomInvert(factor=factor, value_range=value_range, seed=seed, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def randomPerspective(self,factor=None,scale=None,interpolation=None,fill_value=None,seed=None,data_format=None,**kwargs):
        layer=ly.RandomPerspective(factor=factor,scale=scale,interpolation=interpolation,fill_value=fill_value,seed=seed,data_format=data_format,**kwargs)
        self.built[layer.name]=layer
        return layer
    def randomPosterization(self,factor=None, value_range=None, data_format=None, seed=None, **kwargs):
        layer=ly.RandomPosterization(factor=factor, value_range=value_range, data_format=data_format, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    def randomRotation(self,factor=None,fill_mode=None,interpolation=None,seed=None,fill_value=None,data_format=None,**kwargs):
        layer=ly.RandomRotation(factor=factor,fill_mode=fill_mode,interpolation=interpolation,seed=seed,fill_value=fill_value,data_format=data_format,**kwargs)
        self.built[layer.name]=layer
        return layer
    def randomSaturation(self,factor=None, value_range=None, data_format=None, seed=None, **kwargs):
        layer=ly.RandomSaturation(factor=factor, value_range=value_range, data_format=data_format, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    def randomSharpness(self,factor=None, value_range=None, data_format=None, seed=None, **kwargs):
        layer=ly.RandomSharpness(factor=factor, value_range=value_range, data_format=data_format, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    def randomShear(self,x_factor=None,y_factor=None,interpolation=None,fill_mode=None,fill_value=None,data_format=None,seed=None,**kwargs):
        layer=ly.RandomShear(x_factor=x_factor,y_factor=y_factor,interpolation=interpolation,fill_mode=fill_mode,fill_value=fill_value,data_format=data_format,seed=seed,**kwargs)
        self.built[layer.name]=layer
        return layer
    def randomTranslation(self,height_factor=None,width_factor=None,fill_mode=None,interpolation=None,seed=None,fill_value=None,data_format=None,**kwargs):
        layer=ly.RandomTranslation(height_factor=height_factor,width_factor=width_factor,fill_mode=fill_mode,interpolation=interpolation,seed=seed,fill_value=fill_value,data_format=data_format,**kwargs)
        self.built[layer.name]=layer
        return layer
    def randomZoom(self,height_factor=None,width_factor=None,fill_mode=None,interpolation=None,seed=None,fill_value=None,data_format=None,**kwargs):
        layer=ly.RandomZoom(height_factor=height_factor,width_factor=width_factor,fill_mode=fill_mode,interpolation=interpolation,seed=seed,fill_value=fill_value,data_format=data_format,**kwargs)
        self.built[layer.name]=layer
        return layer
    def solarization(self,addition_factor=None, threshold_factor=None, value_range=None, seed=None, **kwargs):
        layer=ly.Solarization(addition_factor=addition_factor, threshold_factor=threshold_factor, value_range=value_range, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    #preprocessing-audio
    def mel_spectrogram(self,fft_length=None,sequence_stride=None,sequence_length=None,window=None,sampling_rate=None,num_mel_bins=None,min_freq=None,max_freq=None,power_to_db=None,top_db=None,mag_exp=None,min_power=None,ref_power=None,**kwargs):
        layer=ly.MelSpectrogram(fft_length=fft_length,sequence_stride=sequence_stride,sequence_length=sequence_length,window=window,sampling_rate=sampling_rate,num_mel_bins=num_mel_bins,min_freq=min_freq,max_freq=max_freq,power_to_db=power_to_db,top_db=top_db,mag_exp=mag_exp,min_power=min_power,ref_power=ref_power,**kwargs)
        self.built[layer.name]=layer
        return layer
    def stftSpectrogram(self,mode=None,frame_length=None,frame_step=None,fft_length=None,window=None,periodic=None,scaling=None,padding=None,expand_dims=None,data_format=None,**kwargs):
        layer=ly.STFTSpectrogram(mode=mode,frame_length=frame_length,frame_step=frame_step,fft_length=fft_length,window=window,periodic=periodic,scaling=scaling,padding=padding,expand_dims=expand_dims,data_format=data_format,**kwargs)
        self.built[layer.name]=layer
        return layer
    #Layer Normalization
    def batchNormalization(self,axis=None,momentum=None,epsilon=None,center=None,scale=None,beta_initializer=None,gamma_initializer=None,moving_mean_initializer=None,moving_variance_initializer=None,beta_regularizer=None,gamma_regularizer=None,beta_constraint=None,gamma_constraint=None,renorm=None,renorm_clipping=None,renorm_momentum=None,synchronized=None,**kwargs):
        layer=ly.BatchNormalization(axis=axis,momentum=momentum,epsilon=epsilon,center=center,scale=scale,beta_initializer=beta_initializer,gamma_initializer=gamma_initializer,moving_mean_initializer=moving_mean_initializer,moving_variance_initializer=moving_variance_initializer,beta_regularizer=beta_regularizer,gamma_regularizer=gamma_regularizer,beta_constraint=beta_constraint,gamma_constraint=gamma_constraint,renorm=renorm,renorm_clipping=renorm_clipping,renorm_momentum=renorm_momentum,synchronized=synchronized,**kwargs)
        self.built[layer.name]=layer
        return layer
    def layerNormalization(self,axis=None,epsilon=None,center=None,scale=None,beta_initializer=None,gamma_initializer=None,beta_regularizer=None,gamma_regularizer=None,beta_constraint=None,gamma_constraint=None,**kwargs):
        layer=ly.LayerNormalization(axis=axis,epsilon=epsilon,center=center,scale=scale,beta_initializer=beta_initializer,gamma_initializer=gamma_initializer,beta_regularizer=beta_regularizer,gamma_regularizer=gamma_regularizer,beta_constraint=beta_constraint,gamma_constraint=gamma_constraint,**kwargs)
        self.built[layer.name]=layer
        return layer
    def unitNormalization(self,axis=None, **kwargs):
        layer=ly.UnitNormalization(axis=axis, **kwargs)
        self.built[layer.name]=layer
        return layer
    def groupNormalization(self,groups=None,axis=None,epsilon=None,center=None,scale=None,beta_initializer=None,gamma_initializer=None,beta_regularizer=None,gamma_regularizer=None,beta_constraint=None,gamma_constraint=None,**kwargs):
        layer=ly.GroupNormalization(groups=groups,axis=axis,epsilon=epsilon,center=center,scale=scale,beta_initializer=beta_initializer,gamma_initializer=gamma_initializer,beta_regularizer=beta_regularizer,gamma_regularizer=gamma_regularizer,beta_constraint=beta_constraint,gamma_constraint=gamma_constraint,**kwargs)
        self.built[layer.name]=layer
        return layer
    def rmsNormalization(self,axis=None, epsilon=None, **kwargs):
        layer=ly.RMSNormalization(axis=axis, epsilon=epsilon, **kwargs)
        self.built[layer.name]=layer
        return layer
    #Regularization
    def dropout(self,rate=None, noise_shape=None, seed=None, **kwargs):
        layer=ly.Dropout(rate=rate, noise_shape=noise_shape, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    def spatialDropout1d(self,rate=None, seed=None, name=None, dtype=None):
        layer=ly.SpatialDropout1D(rate=rate, seed=seed, name=name, dtype=dtype)
        self.built[layer.name]=layer
        return layer
    def spatialDropout2d(self,rate=None, data_format=None, seed=None, name=None, dtype=None):
        layer=ly.SpatialDropout2D(rate=rate, data_format=data_format, seed=seed, name=name, dtype=dtype)
        self.built[layer.name]=layer
        return layer
    def spatialDropout3d(self,rate=None, data_format=None, seed=None, name=None, dtype=None):
        layer=ly.SpatialDropout3D(rate=rate, data_format=data_format, seed=seed, name=name, dtype=dtype)
        self.built[layer.name]=layer
        return layer
    def gaussianDropout(self,rate=None, seed=None, **kwargs):
        layer=ly.GaussianDropout(rate=rate, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    def alphaDropout(self,rate=None, noise_shape=None, seed=None, **kwargs):
        layer=ly.AlphaDropout(rate=rate, noise_shape=noise_shape, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    def gaussianNoise(self,stddev=None, seed=None, **kwargs):
        layer=ly.GaussianNoise(stddev=stddev, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    def activityRegularization(self,l1=None, l2=None, **kwargs):
        layer=ly.ActivityRegularization(l1=l1, l2=l2, **kwargs)
        self.built[layer.name]=layer
        return layer
    #nlp-attention
    def groupQueryAttention(self,head_dim=None,num_query_heads=None,num_key_value_heads=None,dropout=None,use_bias=None,flash_attention=None,kernel_initializer=None,bias_initializer=None,kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,use_gate=None,seed=None,**kwargs):
        layer=ly.GroupQueryAttention(head_dim=head_dim,num_query_heads=num_query_heads,num_key_value_heads=num_key_value_heads,dropout=dropout,use_bias=use_bias,flash_attention=flash_attention,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint,use_gate=use_gate,seed=seed,**kwargs)
        self.built[layer.name]=layer
        return layer
    def multiHeadAttention(self,num_heads=None,key_dim=None,value_dim=None,dropout=None,use_bias=None,output_shape=None,attention_axes=None,flash_attention=None,kernel_initializer=None,bias_initializer=None,kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,use_gate=None,seed=None,**kwargs):
        layer=ly.MultiHeadAttention(num_heads=num_heads,key_dim=key_dim,value_dim=value_dim,dropout=dropout,use_bias=use_bias,output_shape=output_shape,attention_axes=attention_axes,flash_attention=flash_attention,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint,use_gate=use_gate,seed=seed,**kwargs)
        self.built[layer.name]=layer
        return layer
    def attention(self,use_scale=None, score_mode=None, dropout=None, seed=None, **kwargs):
        layer=ly.Attention(use_scale=use_scale, score_mode=score_mode, dropout=dropout, seed=seed, **kwargs)
        self.built[layer.name]=layer
        return layer
    def additiveAttention(self,use_scale=None, dropout=None, **kwargs):
        layer=ly.AdditiveAttention(use_scale=use_scale, dropout=dropout, **kwargs)
        self.built[layer.name]=layer
        return layer
    #Reshaping
    def reshape(self,target_shape=None, **kwargs):
        layer=ly.Reshape(target_shape=target_shape, **kwargs)
        self.built[layer.name]=layer
        return layer
    def flatten(self,data_format=None, **kwargs):
        layer=ly.Flatten(data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def repeatVector(self,n=None,**kwargs):
        layer=ly.RepeatVector(n=n,**kwargs)
        self.built[layer.name]=layer
        return layer
    def permute(self,dims,**kwargs):
        layer=ly.Permute(dims=dims,**kwargs)
        self.built[layer.name]=layer
        return layer
    def cropping1d(self,cropping=None,**kwargs):
        layer=ly.Cropping1D(cropping=cropping,**kwargs)
        self.built[layer.name]=layer
        return layer
    def cropping2d(self,cropping=None, data_format=None, **kwargs):
        layer=ly.Cropping2D(cropping=cropping, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def cropping3d(self,cropping=None, data_format=None, **kwargs):
        layer=ly.Cropping3D(cropping=cropping, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def upSampling1d(self,size=None, **kwargs):
        layer=ly.UpSampling1D(size=size, **kwargs)
        self.built[layer.name]=layer
        return layer
    def upSampling2d(self,size=None, data_format=None, interpolation=None, **kwargs):
        layer=ly.UpSampling2D(size=size, data_format=data_format, interpolation=interpolation, **kwargs)
        self.built[layer.name]=layer
        return layer
    def upSampling3d(self,size=None, data_format=None, **kwargs):
        layer=ly.UpSampling3D(size=size, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def zeroPadding1d(self,padding=None, data_format=None, **kwargs):
        layer=ly.ZeroPadding1D(padding=padding, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def zeroPadding2d(self,padding=None, data_format=None, **kwargs):
        layer=ly.ZeroPadding2D(padding=padding, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    def zeroPadding3d(self,padding=None, data_format=None, **kwargs):
        layer=ly.ZeroPadding3D(padding=padding, data_format=data_format, **kwargs)
        self.built[layer.name]=layer
        return layer
    #merging
    def concatenate(self,axis=None, **kwargs):
        layer=ly.Concatenate(axis=axis, **kwargs)
        self.built[layer.name]=layer
        return layer
    def average(self,**kwargs):
        layer=ly.Average(**kwargs)
        self.built[layer.name]=layer
        return layer
    def maximum(self,**kwargs):
        layer=ly.Maximum(**kwargs)
        self.built[layer.name]=layer
        return layer
    def minimum(self,**kwargs):
        layer=ly.Minimum(**kwargs)
        self.built[layer.name]=layer
        return layer
    def add(self,**kwargs):
        layer=ly.Add(**kwargs)
        self.built[layer.name]=layer
        return layer
    def subtract(self,**kwargs):
        layer=ly.Subtract(**kwargs)
        self.built[layer.name]=layer
        return layer
    def multiply(self,**kwargs):
        layer=ly.Multiply(**kwargs)
        self.built[layer.name]=layer
        return layer
    def dot(self,axes=None, normalize=None, **kwargs):
        layer=ly.Dot(axes=axes, normalize=normalize, **kwargs)
        self.built[layer.name]=layer
        return layer
    #opt-activation
    def reLU(self,max_value=None, negative_slope=None, threshold=None, **kwargs):
        layer=ly.ReLU(max_value=max_value, negative_slope=negative_slope, threshold=threshold, **kwargs)
        self.built[layer.name]=layer
        return layer
    def softMax(self,axis=None, **kwargs):
        layer=ly.Softmax(axis=axis, **kwargs)
        self.built[layer.name]=layer
        return layer
    def leakyReLU(self,negative_slope=None, **kwargs):
        layer=ly.LeakyReLU(negative_slope=negative_slope, **kwargs)
        self.built[layer.name]=layer
        return layer
    def pReLU(self,alpha_initializer=None,alpha_regularizer=None,alpha_constraint=None,shared_axes=None,**kwargs):
        layer=ly.PReLU(alpha_initializer=alpha_initializer,alpha_regularizer=alpha_regularizer,alpha_constraint=alpha_constraint,shared_axes=shared_axes,**kwargs)
        self.built[layer.name]=layer
        return layer
    def elu(self,alpha=None, **kwargs):
        layer=ly.ELU(alpha=alpha, **kwargs)
        self.built[layer.name]=layer
        return layer
    #Backend
    def torchModuleWrapper(self,module=None, name=None, output_shape=None, **kwargs):
        layer=ly.TorchModuleWrapper(module=module, name=name, output_shape=output_shape, **kwargs)
        self.built[layer.name]=layer
        return layer
    def tensorflow_savedModel(self,filepath=None,call_endpoint=None,call_training_endpoint=None,trainable=None,name=None,dtype=None,):
        layer=ly.TFSMLayer(filepath=filepath,call_endpoint=call_endpoint,call_training_endpoint=call_training_endpoint,trainable=trainable,name=name,dtype=dtype,)
        self.built[layer.name]=layer
        return layer
    def jax(self,call_fn=None,init_fn=None,params=None,state=None,seed=None,native_serialization_platforms=None,**kwargs):
        layer=ly.JaxLayer(call_fn=call_fn,init_fn=init_fn,params=params,state=state,seed=seed,native_serialization_platforms=native_serialization_platforms,**kwargs)
        self.built[layer.name]=layer
        return layer
    def flax(self,module=None, method=None, variables=None, **kwargs):
        layer=ly.FlaxLayer(module=module, method=method, variables=variables, **kwargs)
        self.built[layer.name]=layer
        return layer
    