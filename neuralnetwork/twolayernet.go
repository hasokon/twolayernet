package neuralnetwork

import (
	"github.com/hasokon/twolayernet/layers"
	"gonum.org/v1/gonum/mat"
)

type TwoLayerNet struct {
	params           *Params
	affineLayers     []layers.Layer
	activationLayers []layers.ActivationLayer
	lastLayer        layers.OutputLayer
	inputSize        int
	hiddenSize       int
	outputSize       int
	depth            int
}

func InitTwoLayerNet(inputsize, hiddensize, outputsize int, weightInitStd float64) NeuralNetwork {

	t := TwoLayerNet{
		params:     &Params{},
		inputSize:  inputsize,
		hiddenSize: hiddensize,
		outputSize: outputsize,
		depth:      2,
	}

	w1 := makeRandSliceFloat64(inputsize*hiddensize, weightInitStd)
	w2 := makeRandSliceFloat64(hiddensize*outputsize, weightInitStd)
	b1 := makeRandSliceFloat64(hiddensize, weightInitStd)
	b2 := makeRandSliceFloat64(outputsize, weightInitStd)

	t.params.Weight = make([]*mat.Dense, t.depth)
	t.params.Bias = make([]*mat.Dense, t.depth)

	t.params.Weight[0] = mat.NewDense(inputsize, hiddensize, w1)
	t.params.Weight[1] = mat.NewDense(hiddensize, outputsize, w2)
	t.params.Bias[0] = mat.NewDense(1, hiddensize, b1)
	t.params.Bias[1] = mat.NewDense(1, outputsize, b2)

	t.affineLayers = make([]layers.Layer, t.depth)
	t.affineLayers[0] = layers.InitAffineLayer(t.params.Weight[0], t.params.Bias[0])
	t.affineLayers[1] = layers.InitAffineLayer(t.params.Weight[1], t.params.Bias[1])

	t.activationLayers = make([]layers.ActivationLayer, t.depth)
	// t.activationLayers[0] = layers.InitSigmoidLayer()
	t.activationLayers[0] = layers.InitReLuLayer()
	t.activationLayers[1] = layers.InitIdentityLayer()

	t.lastLayer = layers.InitSoftmaxWithLossLayer()

	return &t
}

func (t *TwoLayerNet) Predict(x *mat.Dense) *mat.Dense {
	for i := 0; i < t.depth; i++ {
		x = t.affineLayers[i].Forward(x)
		x = t.activationLayers[i].Forward(x)
	}

	return x
}

func (tl *TwoLayerNet) Loss(x, t *mat.Dense) float64 {
	y := tl.Predict(x)
	return tl.lastLayer.Forward(y, t)
}

func (tl *TwoLayerNet) Accuracy(x, t *mat.Dense) float64 {
	batchSize, _ := x.Dims()

	y := tl.Predict(x)
	sum := 0.0

	for i := 0; i < batchSize; i++ {
		if t.At(i, argmaxOnVec(y.RowView(i))) == 1.0 {
			sum = sum + 1.0
		}
	}

	return sum / float64(batchSize)
}

func (tl *TwoLayerNet) NumericalGradient(x, t *mat.Dense) *Params {
	f := func(w *mat.Dense) float64 {
		return tl.Loss(x, t)
	}

	grads := Params{
		Weight: make([]*mat.Dense, tl.depth),
		Bias:   make([]*mat.Dense, tl.depth),
		Depth:  tl.depth,
	}

	grads.Weight[0] = numericalGradient(f, tl.params.Weight[0])
	grads.Weight[1] = numericalGradient(f, tl.params.Weight[1])
	grads.Bias[0] = numericalGradient(f, tl.params.Bias[0])
	grads.Bias[1] = numericalGradient(f, tl.params.Bias[1])

	return &grads
}

func (tl *TwoLayerNet) Gradient(x, t *mat.Dense) *Params {
	tl.Loss(x, t)

	dout := tl.lastLayer.Backward(1.0)

	for i := tl.depth - 1; i >= 0; i-- {
		dout = tl.activationLayers[i].Backward(dout)
		dout = tl.affineLayers[i].Backward(dout)
	}

	weight := make([]*mat.Dense, tl.depth)
	bias := make([]*mat.Dense, tl.depth)
	for i := 0; i < tl.depth; i++ {
		weight[i] = tl.affineLayers[i].GetDW()
		bias[i] = tl.affineLayers[i].GetDB()
	}

	grads := Params{
		Weight: weight,
		Bias:   bias,
	}

	return &grads
}

func (tl *TwoLayerNet) GetParams() *Params {
	return tl.params
}

func (tl *TwoLayerNet) GetDepth() int {
	return tl.depth
}
