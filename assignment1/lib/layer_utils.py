import numpy as np


class sequential(object):
	def __init__(self, *args):
		"""
		Sequential Object to serialize the NN layers
		Please read this code block and understand how it works
		"""
		self.params = {}
		self.grads = {}
		self.layers = []
		self.paramName2Indices = {}
		self.layer_names = {}

		# process the parameters layer by layer
		layer_cnt = 0
		for layer in args:
			for n, v in layer.params.iteritems():
				if v is None:
					continue
				self.params[n] = v
				self.paramName2Indices[n] = layer_cnt
			for n, v in layer.grads.iteritems():
				self.grads[n] = v
			if layer.name in self.layer_names:
				raise ValueError("Existing name {}!".format(layer.name))
			self.layer_names[layer.name] = True
			self.layers.append(layer)
			layer_cnt += 1
		layer_cnt = 0

	def assign(self, name, val):
		# load the given values to the layer by name
		layer_cnt = self.paramName2Indices[name]
		self.layers[layer_cnt].params[name] = val

	def assign_grads(self, name, val):
		# load the given values to the layer by name
		layer_cnt = self.paramName2Indices[name]
		self.layers[layer_cnt].grads[name] = val

	def get_params(self, name):
		# return the parameters by name
		return self.params[name]

	def get_grads(self, name):
		# return the gradients by name
		return self.grads[name]

	def gather_params(self):
		"""
		Collect the parameters of every submodules
		"""
		for layer in self.layers:
			for n, v in layer.params.iteritems():
				self.params[n] = v

	def gather_grads(self):
		"""
		Collect the gradients of every submodules
		"""
		for layer in self.layers:
			for n, v in layer.grads.iteritems():
				self.grads[n] = v

	def load(self, pretrained):
		"""
		Load a pretrained model by names
		"""
		for layer in self.layers:
			if not hasattr(layer, "params"):
				continue
			for n, v in layer.params.iteritems():
				if n in pretrained.keys():
					layer.params[n] = pretrained[n].copy()
					print "Loading Params: {} Shape: {}".format(n, layer.params[n].shape)


class fc(object):
	def __init__(self, input_dim, output_dim, init_scale=0.02, name="fc"):
		"""
		In forward pass, please use self.params for the weights and biases for this layer
		In backward pass, store the computed gradients to self.grads
		- name: the name of current layer
		- input_dim: input dimension
		- output_dim: output dimension
		- meta: to store the forward pass activations for computing backpropagation
		"""
		self.name = name
		self.w_name = name + "_w"
		self.b_name = name + "_b"
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.params = {}
		self.grads = {}
		self.params[self.w_name] = init_scale * np.random.randn(input_dim, output_dim)
		self.params[self.b_name] = np.zeros(output_dim)
		self.grads[self.w_name] = None
		self.grads[self.b_name] = None
		self.meta = None

	def forward(self, feat):
		""" Some comments """
		output = None
		assert np.prod(feat.shape[1:]) == self.input_dim, "But got {} and {}".format(
			np.prod(feat.shape[1:]), self.input_dim)

		#############################################################################
		# TODO: Implement the forward pass of a single fully connected layer.       #
		# You will probably need to reshape (flatten) the input features.           #
		# Store the results in the variable output provided above.                  #
		#############################################################################

		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.meta = feat  
		D = np.prod(self.meta.shape[1:])
		input_ = self.meta.reshape(self.meta.shape[0], D)
		output = np.dot(input_, self.params[self.w_name]) + self.params[self.b_name]
		return output

	def backward(self, dprev):
		""" Some comments """
		feat = self.meta
		if feat is None:
			raise ValueError("No forward function called before for this module!")
		dfeat, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
		self.grads[self.w_name] = self.meta.reshape(self.meta.shape[0], self.params[self.w_name].shape[0]).T.dot(dprev)
		self.grads[self.b_name] = np.sum(dprev, axis=0)
		dfeat = dprev.dot(self.params[self.w_name].T).reshape(self.meta.shape)

		#############################################################################
		# TODO: Implement the backward pass of a single fully connected layer.      #
		# You will probably need to reshape (flatten) the input gradients.          #
		# Store the computed gradients for current layer in self.grads with         #
		# corresponding name.                                                       #
		#############################################################################

		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.meta = None
		return dfeat


class relu(object):
	def __init__(self, name="relu"):
		"""
		- name: the name of current layer
		Note: params and grads should be just empty dicts here, do not update them
		"""
		self.name = name
		self.params = {}
		self.grads = {}
		self.grads[self.name] = None
		self.meta = None

	def forward(self, feat):
		""" Some comments """
		output = None
		#############################################################################
		# TODO: Implement the forward pass of a rectified linear unit               #
		# Store the results in the variable output provided above.                  #
		#############################################################################
		output = np.maximum(0, feat)
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.meta = feat
		return output

	def backward(self, dprev):
		""" Some comments """
		feat = self.meta
		if feat is None:
			raise ValueError("No forward function called before for this module!")
		dfeat = None
		#############################################################################
		# TODO: Implement the backward pass of a rectified linear unit              #
		#############################################################################
		dfeat = dprev
		dfeat[feat < 0] = 0
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.meta = None
		return dfeat


class dropout(object):
	def __init__(self, p, seed=None, name="dropout"):
		"""
		- name: the name of current layer
		- p: the dropout probability
		- seed: numpy random seed
		- meta: to store the forward pass activations for computing backpropagation
		- dropped: the mask for dropping out the neurons
		- is_Training: dropout behaves differently during training and testing, use
		               this to indicate which phase is the current one
		"""
		self.name = name
		self.params = {}
		self.grads = {}
		self.grads[self.name] = None
		self.p = p
		self.seed = seed
		self.meta = None
		self.dropped = None
		self.is_Training = False

	def forward(self, feat, is_Training=True):
		if self.seed is not None:
			np.random.seed(self.seed)
		dropped = None
		output = None
        
		if is_Training == True and self.p > 0:
				dropped = (np.random.rand(*feat.shape) < self.p) / self.p
				output = feat * dropped
       
		else:
				dropped = None
				output = feat
		output = output.astype(feat.dtype, copy=False)
		#############################################################################
		# TODO: Implement the forward pass of Dropout                               #
		#############################################################################

		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.dropped = dropped
		self.is_Training = is_Training
		self.meta = feat
		return output

	def backward(self, dprev):
		feat = self.meta
		dfeat = None
		if feat is None:
			raise ValueError("No forward function called before for this module!")
		#############################################################################
		# TODO: Implement the backward pass of Dropout                              #
		#############################################################################
		if self.is_Training == True and self.p > 0:
				dfeat = dprev * self.dropped
		else:
				dfeat = dprev
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.is_Training = False
		return dfeat


class cross_entropy(object):
	def __init__(self, dim_average=True):
		"""
		- dim_average: if dividing by the input dimension or not
		- dLoss: intermediate variables to store the scores
		- label: Ground truth label for classification task
		"""
		self.dim_average = dim_average  # if average w.r.t. the total number of features
		self.dLoss = None
		self.label = None

	def forward(self, feat, label):
		""" Some comments """
		scores = softmax(feat)   
		loss = None
		#############################################################################
		# TODO: Implement the forward pass of an CE Loss                            #
		#############################################################################
		N = feat.shape[0]
		#print(scores[np.arange(N), label])
		#print(np.log(scores[np.arange(N), label]))        
		loss = -np.sum(np.log(scores[np.arange(N), label])) / N 
		self.dLoss = scores.copy()
		self.label = label
		return loss

	def backward(self):
		dLoss = self.dLoss
		if dLoss is None:
			raise ValueError("No forward function called before for this module!")
		#############################################################################
		# TODO: Implement the backward pass of an CE Loss                           #
		#############################################################################
		dLoss[np.arange(self.label.shape[0]), self.label] -= 1
		dLoss /= self.label.shape[0]
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.dLoss = dLoss
		return dLoss


def softmax(feat):
	""" Some comments """
	scores = None
	#############################################################################
	# TODO: Implement the forward pass of a softmax function                    #
	#############################################################################
	scores = np.exp(feat - np.max(feat, axis=1, keepdims=True))
	scores /= np.sum(scores, axis=1, keepdims=True)
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return scores
