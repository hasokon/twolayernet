package layers

import (
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	Forward(*mat.Dense) *mat.Dense
	Backward(*mat.Dense) *mat.Dense
	GetDB() *mat.Dense
	GetDW() *mat.Dense
}

type ActivationLayer interface {
	Forward(*mat.Dense) *mat.Dense
	Backward(*mat.Dense) *mat.Dense
}

type OutputLayer interface {
	Forward(x, t *mat.Dense) float64
	Backward(dout float64) *mat.Dense
}
