package neuralnetwork

import (
	"math"
	"time"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

type NeuralNetwork interface {
	Predict(input *mat.Dense) *mat.Dense
	Loss(x, t *mat.Dense) float64
	Accuracy(x, t *mat.Dense) float64
	NumericalGradient(x, t *mat.Dense) *Params
	Gradient(x, t *mat.Dense) *Params
	GetParams() *Params
	GetDepth() int
}

func argmaxOnVec(v mat.Vector) int {
	len := v.Len()
	max := v.AtVec(0)
	argmax := 0
	for i := 1; i < len; i++ {
		if v.AtVec(i) > max {
			max = v.AtVec(i)
			argmax = i
		}
	}

	return argmax
}

func makeRandSliceFloat64(size int, param float64) []float64 {
	slc := make([]float64, size)
	rand.Seed(uint64(time.Now().UnixNano()))
	for i := 0; i < size; i++ {
		slc[i] = rand.NormFloat64() * param
		if rand.Intn(2) == 0 {
			slc[i] = slc[i] * -1.0
		}
	}
	return slc
}

func makeRandSliceOneHot(size int) []float64 {
	slc := make([]float64, size)
	rand.Seed(uint64(time.Now().UnixNano()))
	for i := 0; i < size; i++ {
		slc[i] = float64(rand.Int())
	}
	return slc
}

func numericalGradient(f func(*mat.Dense) float64, x *mat.Dense) *mat.Dense {
	h := math.Pow10(-4)
	r, c := x.Dims()

	grad := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			tmpval := x.At(i, j)
			x.Set(i, j, tmpval-h)
			fx0 := f(x)
			x.Set(i, j, tmpval+h)
			fx1 := f(x)

			grad.Set(i, j, (fx1-fx0)/(2*h))
			x.Set(i, j, tmpval)
		}
	}

	return grad
}
