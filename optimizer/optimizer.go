package optimizer

import "github.com/hasokon/twolayernet/neuralnetwork"

const (
	AlgorismSGD = iota
	AlgorismMomentum
	AlgorismAdaGrad
)

type Algorism int

type Optimizer interface {
	Update(params, grads *neuralnetwork.Params)
}

func InitOptimizer(depth int, learningRate float64, a Algorism) Optimizer {
	switch a {
	case AlgorismSGD:
		return InitSGD(learningRate)
	case AlgorismMomentum:
		return InitMomentum(depth, learningRate, 0.9)
	case AlgorismAdaGrad:
		return InitAdaGrad(depth, learningRate)
	}

	return InitSGD(learningRate)
}
