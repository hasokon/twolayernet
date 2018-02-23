package main

import (
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

func makeRandSliceFloat64(size int, param float64) []float64 {
	slc := make([]float64, size)
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < size; i++ {
		slc[i] = rand.NormFloat64() * param
	}
	return slc
}

func makeRandSliceInt(size int) []float64 {
	slc := make([]float64, size)
	rand.Seed(time.Now().UnixNano())
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

			grad.Set(i, j, (fx0-fx1)/(2*h))
			x.Set(i, j, tmpval)
		}
	}

	return grad
}

type Params struct {
	Weight       []*mat.Dense
	Bias         []*mat.Dense
	InputSize    int
	HiddenSize   int
	OutputSize   int
	LearningRate float64
}

type TwoLayerNet interface {
	Predict(input *mat.Dense) *mat.Dense
	Loss(x, t *mat.Dense) float64
	Accuracy(x, t *mat.Dense) float64
	NumericalGradient(x, t *mat.Dense) *Params
	ParamUpdate(p *Params)
}

func InitNet(inputsize, hiddensize, outputsize int, weightInitStd, learningRate float64) TwoLayerNet {
	p := Params{
		Weight:       make([]*mat.Dense, 2),
		Bias:         make([]*mat.Dense, 2),
		InputSize:    inputsize,
		HiddenSize:   hiddensize,
		OutputSize:   outputsize,
		LearningRate: learningRate,
	}

	w1 := makeRandSliceFloat64(inputsize*hiddensize, weightInitStd)
	w2 := makeRandSliceFloat64(hiddensize*outputsize, weightInitStd)
	b1 := makeRandSliceFloat64(hiddensize, weightInitStd)
	b2 := makeRandSliceFloat64(outputsize, weightInitStd)

	p.Weight[0] = mat.NewDense(inputsize, hiddensize, w1)
	p.Weight[1] = mat.NewDense(hiddensize, outputsize, w2)
	p.Bias[0] = mat.NewDense(1, hiddensize, b1)
	p.Bias[1] = mat.NewDense(1, outputsize, b2)

	return &p
}

func (p *Params) Predict(input *mat.Dense) *mat.Dense {

	batchSize, _ := input.Dims()

	// Input -> Hidden
	a1 := mat.NewDense(batchSize, p.HiddenSize, nil)
	a1.Mul(input, p.Weight[0])
	for i := 0; i < batchSize; i++ {
		tmprow := a1.RawRowView(i)
		for j, x := range tmprow {
			tmprow[j] = x + p.Bias[0].At(0, j)
		}
		a1.SetRow(i, tmprow)
	}
	z1 := sigmoid(a1)

	// Hidden -> Output
	a2 := mat.NewDense(batchSize, p.OutputSize, nil)
	a2.Mul(z1, p.Weight[1])
	for i := 0; i < batchSize; i++ {
		tmprow := a2.RawRowView(i)
		for j, x := range tmprow {
			tmprow[j] = x + p.Bias[1].At(0, j)
		}
		a2.SetRow(i, tmprow)
	}
	output := softmax(a2)

	return output
}

func (p *Params) Loss(x, t *mat.Dense) float64 {
	y := p.Predict(x)
	return crossEntropyError(y, t)
}

func (p *Params) Accuracy(x, t *mat.Dense) float64 {
	batchSize, _ := x.Dims()

	y := p.Predict(x)
	sum := 0.0

	for i := 0; i < batchSize; i++ {
		if t.At(i, argmaxOnVec(y.RowView(i))) == 1.0 {
			sum = sum + 1.0
		}
	}

	return sum / float64(batchSize)
}

func (p *Params) NumericalGradient(x, t *mat.Dense) *Params {
	f := func(w *mat.Dense) float64 {
		return p.Loss(x, t)
	}

	grads := Params{
		Weight: make([]*mat.Dense, 2),
		Bias:   make([]*mat.Dense, 2),
	}

	grads.Weight[0] = numericalGradient(f, p.Weight[0])
	grads.Weight[1] = numericalGradient(f, p.Weight[1])
	grads.Bias[0] = numericalGradient(f, p.Bias[0])
	grads.Bias[1] = numericalGradient(f, p.Bias[1])

	return &grads
}

func (p *Params) ParamUpdate(param *Params) {
	for d := 0; d < 2; d++ {
		r, c := param.Weight[d].Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				param.Weight[d].Set(i, j, param.Weight[d].At(i, j)*p.LearningRate)
			}
		}
		r, c = param.Bias[d].Dims()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				param.Bias[d].Set(i, j, param.Bias[d].At(i, j)*p.LearningRate)
			}
		}
	}

	p.Weight[0].Add(p.Weight[0], param.Weight[0])
	p.Weight[1].Add(p.Weight[1], param.Weight[1])
	p.Bias[0].Add(p.Bias[0], param.Bias[0])
	p.Bias[1].Add(p.Bias[1], param.Bias[1])
}
