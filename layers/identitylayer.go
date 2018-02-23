package layers

import (
	"gonum.org/v1/gonum/mat"
)

type IdentityLayer struct{}

func InitIdentityLayer() *IdentityLayer {
	return &IdentityLayer{}
}

func (i *IdentityLayer) Forward(x *mat.Dense) *mat.Dense {
	return x
}

func (i *IdentityLayer) Backward(dout *mat.Dense) *mat.Dense {
	return dout
}
