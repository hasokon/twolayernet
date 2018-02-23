package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func softmax(x *mat.Dense) *mat.Dense {
	r, c := x.Dims()
	ans := mat.NewDense(r, c, nil)

	max := mat.Max(x)

	for i := 0; i < r; i++ {
		sum := 0.0
		for j := 0; j < c; j++ {
			ans.Set(i, j, math.Exp(x.At(i, j)-max))
			sum = sum + ans.At(i, j)
		}
		for j := 0; j < c; j++ {
			ans.Set(i, j, ans.At(i, j)/sum)
		}
	}

	return ans
}

func crossEntropyError(x, t *mat.Dense) float64 {
	delta := math.Pow10(-7)
	batchSize, outputSize := x.Dims()

	sum := 0.0

	for i := 0; i < batchSize; i++ {
		for j := 0; j < outputSize; j++ {
			if t.At(i, j) == 1.0 {
				sum = sum + math.Log(x.At(i, j)+delta)
				break
			}
		}
	}

	return -1.0 * sum / float64(batchSize)
}

type SoftmaxWithLossLayer struct {
	loss float64
	y    *mat.Dense
	t    *mat.Dense
}

func InitSoftmaxWithLossLayer() *SoftmaxWithLossLayer {
	return &SoftmaxWithLossLayer{}
}

func (s *SoftmaxWithLossLayer) Forward(x, t *mat.Dense) float64 {
	s.t = t
	s.y = softmax(x)
	s.loss = crossEntropyError(s.y, s.t)

	return s.loss
}

func (s *SoftmaxWithLossLayer) Backward(dout float64) *mat.Dense {
	r, c := s.t.Dims()
	dx := mat.NewDense(r, c, nil)

	batchSize := float64(r)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			dx.Set(i, j, (s.y.At(i, j)-s.t.At(i, j))/batchSize)
		}
	}

	return dx
}
