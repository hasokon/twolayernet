package layers

import (
	"gonum.org/v1/gonum/mat"
)

type AffineLayer struct {
	W  *mat.Dense
	B  *mat.Dense
	X  *mat.Dense
	DW *mat.Dense
	DB *mat.Dense
}

func InitAffineLayer(w, b *mat.Dense) *AffineLayer {
	return &AffineLayer{
		W: w,
		B: b,
	}
}

func (a *AffineLayer) GetDB() *mat.Dense {
	return a.DB
}

func (a *AffineLayer) GetDW() *mat.Dense {
	return a.DW
}

func (a *AffineLayer) Forward(x *mat.Dense) *mat.Dense {
	a.X = x
	batchSize, _ := x.Dims()
	_, layerSize := a.W.Dims()

	out := mat.NewDense(batchSize, layerSize, nil)
	out.Mul(x, a.W)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < layerSize; j++ {
			out.Set(i, j, out.At(i, j)+a.B.At(0, j))
		}
	}

	return out
}

func (a *AffineLayer) Backward(dout *mat.Dense) *mat.Dense {
	wt := a.W.T()
	rdout, cdout := dout.Dims()
	_, cwt := wt.Dims()

	dx := mat.NewDense(rdout, cwt, nil)
	dx.Mul(dout, wt)

	xt := a.X.T()
	rxt, _ := xt.Dims()
	a.DW = mat.NewDense(rxt, cdout, nil)
	a.DW.Mul(xt, dout)

	_, cb := a.B.Dims()
	a.DB = mat.NewDense(1, cb, nil)

	for i := 0; i < rdout; i++ {
		for j := 0; j < cdout; j++ {
			a.DB.Set(0, j, a.DB.At(0, j)+dout.At(i, j))
		}
	}

	return dx
}
