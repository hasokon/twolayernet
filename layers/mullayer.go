package layers

import (
	"gonum.org/v1/gonum/mat"
)

type MulLayer struct {
	x *mat.Dense
	y *mat.Dense
}

func InitMulLayer() *MulLayer {
	return &MulLayer{}
}

func (m *MulLayer) Forward(x, y *mat.Dense) *mat.Dense {
	m.x = mat.DenseCopyOf(x)
	m.y = mat.DenseCopyOf(y)

	r, c := x.Dims()

	out := mat.NewDense(r, c, nil)
	out.MulElem(x, y)
	return out
}

func (m *MulLayer) Backward(dout *mat.Dense) (dx, dy *mat.Dense) {
	r, c := dout.Dims()

	dx = mat.NewDense(r, c, nil)
	dx.MulElem(dout, m.y)
	dy = mat.NewDense(r, c, nil)
	dy.MulElem(dout, m.x)

	return
}
