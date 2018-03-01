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
