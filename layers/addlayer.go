package layers

import (
	"gonum.org/v1/gonum/mat"
)

type AddLayer struct{}

func InitAddLayer() *AddLayer {
	return &AddLayer{}
}

func (a *AddLayer) Forward(x, y *mat.Dense) *mat.Dense {
	r, c := x.Dims()
	out := mat.NewDense(r, c, nil)
	out.Add(x, y)
	return out
}

func (a *AddLayer) Backward(dout *mat.Dense) (dx, dy *mat.Dense) {
	dx = mat.DenseCopyOf(dout)
	dy = mat.DenseCopyOf(dout)
	return
}
