package optimizer

import "github.com/hasokon/twolayernet/neuralnetwork"

type SGD struct {
	learningRate float64
}

func InitSGD(learningrate float64) Optimizer {
	return &SGD{
		learningRate: learningrate,
	}
}

func (s *SGD) Update(params, grads *neuralnetwork.Params) {
	for d := 0; d < 2; d++ {
		r, c := grads.Weight[d].Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				grads.Weight[d].Set(i, j, grads.Weight[d].At(i, j)*s.learningRate*-1.0)
			}
		}
		r, c = grads.Bias[d].Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				grads.Bias[d].Set(i, j, grads.Bias[d].At(i, j)*s.learningRate*-1.0)
			}
		}
	}

	params.Weight[0].Add(params.Weight[0], grads.Weight[0])
	params.Weight[1].Add(params.Weight[1], grads.Weight[1])
	params.Bias[0].Add(params.Bias[0], grads.Bias[0])
	params.Bias[1].Add(params.Bias[1], grads.Bias[1])
}
