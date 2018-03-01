package layers

import "gonum.org/v1/gonum/mat"

type ActivationLayer interface {
	Forward(*mat.Dense) *mat.Dense
	Backward(*mat.Dense) *mat.Dense
}
