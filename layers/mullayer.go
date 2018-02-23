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

	r, _ := x.Dims()
	_, c := y.Dims()

	out := mat.NewDense(r, c, nil)
	out.Mul(x, y)
	return out
}

func (m *MulLayer) Backward(dout *mat.Dense) (dx, dy *mat.Dense) {
	r, _ := dout.Dims()
	_, cdx := m.y.Dims()
	_, cdy := m.y.Dims()

	dx = mat.NewDense(r, cdx, nil)
	dx.Mul(dout, m.y)
	dy = mat.NewDense(r, cdy, nil)
	dy.Mul(dout, m.x)

	return
}
