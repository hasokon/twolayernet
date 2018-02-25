package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type SigmoidLayer struct {
	out *mat.Dense
}

func InitSigmoidLayer() ActivationLayer {
	return &SigmoidLayer{}
}

func (s *SigmoidLayer) Forward(x *mat.Dense) *mat.Dense {
	r, c := x.Dims()
	out := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out.Set(i, j, (1 / (1 + math.Exp(-1.0*x.At(i, j)))))
		}
	}
	s.out = mat.DenseCopyOf(out)
	return out
}

func (s *SigmoidLayer) Backward(dout *mat.Dense) (dx *mat.Dense) {
	r, c := dout.Dims()
	dx = mat.NewDense(r, c, nil)
	y := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			y = s.out.At(i, j)
			dx.Set(i, j, dout.At(i, j)*(1-y)*y)
		}
	}
	return dx
}
