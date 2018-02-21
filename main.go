package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func sigmoid(x *mat.Dense) *mat.Dense {
	r, c := x.Dims()
	ans := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			ans.Set(i, j, (1.0 / (1.0 + math.Exp(-1.0*x.At(i, j)))))
		}
	}

	return ans
}

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

func makeRandSliceFloat64(size int) []float64 {
	slc := make([]float64, size)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < size; i++ {
		slc[i] = rand.Float64()
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
			x.Set(i, j, tmpval+h)
			fx0 := f(x)
			x.Set(i, j, tmpval-h)
			fx1 := f(x)

			grad.Set(i, j, (fx0-fx1)/(2*h))
			x.Set(i, j, tmpval)
		}
	}

	return grad
}

type TwoLayerNet struct {
	weight     []*mat.Dense
	bias       []*mat.Dense
	inputSize  int
	hiddenSize int
	outputSize int
}

func InitNet(inputsize, hiddensize, outputsize int, weightInitStd float64) *TwoLayerNet {
	tln := TwoLayerNet{
		weight:     make([]*mat.Dense, 2),
		bias:       make([]*mat.Dense, 2),
		inputSize:  inputsize,
		hiddenSize: hiddensize,
		outputSize: outputsize,
	}

	w1 := makeRandSliceFloat64(inputsize * hiddensize)
	w2 := makeRandSliceFloat64(hiddensize * outputsize)
	b1 := makeRandSliceFloat64(hiddensize)
	b2 := makeRandSliceFloat64(outputsize)

	tln.weight[0] = mat.NewDense(inputsize, hiddensize, w1)
	tln.weight[1] = mat.NewDense(hiddensize, outputsize, w2)
	tln.bias[0] = mat.NewDense(1, hiddensize, b1)
	tln.bias[1] = mat.NewDense(1, outputsize, b2)

	return &tln
}

func (tln *TwoLayerNet) Predict(input *mat.Dense) *mat.Dense {

	batchSize, _ := input.Dims()

	// Input -> Hidden
	a1 := mat.NewDense(batchSize, tln.hiddenSize, nil)
	a1.Mul(input, tln.weight[0])
	for i := 0; i < batchSize; i++ {
		tmprow := a1.RawRowView(i)
		for j, x := range tmprow {
			tmprow[j] = x + tln.bias[0].At(0, j)
		}
		a1.SetRow(i, tmprow)
	}
	z1 := sigmoid(a1)

	// Hidden -> Output
	a2 := mat.NewDense(batchSize, tln.outputSize, nil)
	a2.Mul(z1, tln.weight[1])
	for i := 0; i < batchSize; i++ {
		tmprow := a2.RawRowView(i)
		for j, x := range tmprow {
			tmprow[j] = x + tln.bias[1].At(0, j)
		}
		a2.SetRow(i, tmprow)
	}
	output := softmax(a2)

	return output
}

func (tln *TwoLayerNet) Loss(x, t *mat.Dense) float64 {
	y := tln.Predict(x)
	return crossEntropyError(y, t)
}

func (tln *TwoLayerNet) Accuracy(x, t *mat.Dense) float64 {
	batchSize, _ := x.Dims()

	y := tln.Predict(x)
	sum := 0.0

	for i := 0; i < batchSize; i++ {
		if t.At(i, argmaxOnVec(y.RowView(i))) == 1.0 {
			sum = sum + 1.0
		}
	}

	return sum / float64(batchSize)
}

// func (tln *TwoLayerNet) NumericalGradient(x, t *mat.Dense) *mat.Dense {
// 	f := func(w *mat.Dense) float64 {
// 		return tln.Loss(x, t)
// 	}

// 	grads := TwoLayerNet{
// 		weight: make([]*mat.Dense, 2),
// 		bias:   make([]*mat.Dense, 2),
// 	}
// }

func main() {
	is := 784
	hs := 50
	os := 10
	bs := 1

	net := InitNet(is, hs, os, 0)

	input := mat.NewDense(bs, is, makeRandSliceFloat64(bs*is))
	t := mat.NewDense(bs, os, nil)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < bs; i++ {
		truearg := rand.Intn(10)
		t.Set(i, truearg, 1.0)
	}

	fmt.Println(mat.Formatted(t))
	fmt.Println(mat.Formatted(net.Predict(input)))

	fmt.Println(net.Loss(input, t))
	fmt.Println(net.Accuracy(input, t))
}
