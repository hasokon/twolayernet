package layers

import (
	"gonum.org/v1/gonum/mat"
)

type ReLuLayer struct {
	mask [][]bool
}

func InitReLuLayer() ActivationLayer {
	return &ReLuLayer{}
}

func (r *ReLuLayer) Forward(x *mat.Dense) *mat.Dense {
	rows, cols := x.Dims()
	out := mat.NewDense(rows, cols, nil)
	r.mask = make([][]bool, rows)
	for i := 0; i < rows; i++ {
		r.mask[i] = make([]bool, cols)
		for j := 0; j < cols; j++ {
			v := x.At(i, j)
			if v > 0 {
				r.mask[i][j] = true
				out.Set(i, j, v)
			} else {
				r.mask[i][j] = false
				out.Set(i, j, 0)
			}
		}
	}

	return out
}

func (r *ReLuLayer) Backward(dout *mat.Dense) *mat.Dense {
	rows, cols := dout.Dims()
	dx := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if r.mask[i][j] {
				dx.Set(i, j, dout.At(i, j))
			}
		}
	}

	return dx
}
