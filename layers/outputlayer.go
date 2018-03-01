package layers

import "gonum.org/v1/gonum/mat"

type OutputLayer interface {
	Forward(x, t *mat.Dense) float64
	Backward(dout float64) *mat.Dense
}
