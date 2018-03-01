package layers

import (
	"gonum.org/v1/gonum/mat"
)

type NoNormalizationLayer struct{}

func InitNoNormalizationLayer(b, g []float64) NormalizationLayer {
	return &NoNormalizationLayer{}
}

func (n *NoNormalizationLayer) Forward(x *mat.Dense) *mat.Dense {
	return x
}

func (n *NoNormalizationLayer) Backward(dout *mat.Dense) *mat.Dense {
	return dout
}
