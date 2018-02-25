package optimizer

import (
	"math"

	"github.com/hasokon/twolayernet/neuralnetwork"
	"gonum.org/v1/gonum/mat"
)

type AdaGrad struct {
	learningRate float64
	h            *neuralnetwork.Params
}

func InitAdaGrad(depth int, learningrate float64) Optimizer {
	return &AdaGrad{
		learningRate: learningrate,
		h:            neuralnetwork.InitParams(depth),
	}
}

func (a *AdaGrad) Update(params, grads *neuralnetwork.Params) {

	delta := math.Pow10(-7)

	for d := 0; d < a.h.Depth; d++ {
		// Weight
		r, c := params.Weight[d].Dims()
		if a.h.Weight[d] == nil {
			a.h.Weight[d] = mat.NewDense(r, c, nil)
		}

		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				grad := grads.Weight[d].At(i, j)
				h := a.h.Weight[d].At(i, j) + grad*grad
				a.h.Weight[d].Set(i, j, h)
				params.Weight[d].Set(i, j, params.Weight[d].At(i, j)-a.learningRate*grad/(math.Sqrt(h)+delta))
			}
		}

		// Bias
		r, c = params.Bias[d].Dims()
		if a.h.Bias[d] == nil {
			a.h.Bias[d] = mat.NewDense(r, c, nil)
		}

		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				grad := grads.Bias[d].At(i, j)
				h := a.h.Bias[d].At(i, j) + grad*grad
				a.h.Bias[d].Set(i, j, h)
				params.Bias[d].Set(i, j, params.Bias[d].At(i, j)-a.learningRate*grad/(math.Sqrt(h)+delta))
			}
		}
	}
}
