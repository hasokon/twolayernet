package neuralnetwork

import (
	"gonum.org/v1/gonum/mat"
)

type Params struct {
	Weight []*mat.Dense
	Bias   []*mat.Dense
	Depth  int
}

func InitParams(depth int) *Params {
	return &Params{
		Weight: make([]*mat.Dense, depth),
		Bias:   make([]*mat.Dense, depth),
		Depth:  depth,
	}
}
