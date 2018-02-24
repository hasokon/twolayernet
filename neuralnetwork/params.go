package neuralnetwork

import (
	"gonum.org/v1/gonum/mat"
)

type Params struct {
	Weight []*mat.Dense
	Bias   []*mat.Dense
	Depth  int
}
