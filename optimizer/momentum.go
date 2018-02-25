package optimizer

import (
	"github.com/hasokon/twolayernet/neuralnetwork"
	"gonum.org/v1/gonum/mat"
)

type Momentum struct {
	learningRate float64
	momentum     float64
	v            *neuralnetwork.Params
}

func InitMomentum(depth int, learningrate, momentum float64) Optimizer {
	return &Momentum{
		learningRate: learningrate,
		momentum:     momentum,
		v:            neuralnetwork.InitParams(depth),
	}
}

func (m *Momentum) Update(params, grads *neuralnetwork.Params) {
	for d := 0; d < m.v.Depth; d++ {
		// Weight
		r, c := params.Weight[d].Dims()
		if m.v.Weight[d] == nil {
			m.v.Weight[d] = mat.NewDense(r, c, nil)
		}
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				p := m.momentum*m.v.Weight[d].At(i, j) - grads.Weight[d].At(i, j)*m.learningRate
				m.v.Weight[d].Set(i, j, p)
				params.Weight[d].Set(i, j, params.Weight[d].At(i, j)+p)
			}
		}

		// Bias
		r, c = params.Bias[d].Dims()
		if m.v.Bias[d] == nil {
			m.v.Bias[d] = mat.NewDense(r, c, nil)
		}
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				p := m.momentum*m.v.Bias[d].At(i, j) - grads.Bias[d].At(i, j)*m.learningRate
				m.v.Bias[d].Set(i, j, p)
				params.Bias[d].Set(i, j, params.Bias[d].At(i, j)+p)
			}
		}
	}
}
