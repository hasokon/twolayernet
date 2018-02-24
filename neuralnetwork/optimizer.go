package neuralnetwork

type Optimizer interface {
	Update(params, grads *Params)
}
