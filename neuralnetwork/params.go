package neuralnetwork

import (
	"gonum.org/v1/gonum/mat"
)

type Params struct {
	Weight []*mat.Dense
	Bias   []*mat.Dense
	Beta   [][]float64
	Gamma  [][]float64
	Depth  int
}

func InitParams(depth int) *Params {
	return &Params{
		Weight: make([]*mat.Dense, depth),
		Bias:   make([]*mat.Dense, depth),
		Beta:   make([][]float64, depth),
		Gamma:  make([][]float64, depth),
		Depth:  depth,
	}
}
