package neuralnetwork

import (
	"errors"

	"github.com/hasokon/twolayernet/layers"
	"gonum.org/v1/gonum/mat"
)

const (
	ActivationAlgorismSigmoid = iota
	ActivationAlgorismReLu
)

const (
	NormalizationAlgorismNo = iota
	NormalizationAlgorismBatchNorm
)

type ActivationAlgorism int
type NormalizationAlgorism int

type MultiLayerNet struct {
	params              *Params
	affineLayers        []layers.Layer
	activationLayers    []layers.ActivationLayer
	normalizationLayers []layers.NormalizationLayer
	lastLayer           layers.OutputLayer
	neurons             []int
	depth               int
}

func InitMultiLayerNet(neurons []int, weightInitStd float64, a ActivationAlgorism, n NormalizationAlgorism) (NeuralNetwork, error) {

	depth := len(neurons) - 1

	if depth < 1 {
		return nil, errors.New("Invalid Args: the length of neurons is at least 2")
	}

	m := MultiLayerNet{
		params:  InitParams(depth),
		neurons: neurons,
		depth:   depth,
	}

	m.affineLayers = make([]layers.Layer, m.depth)
	m.activationLayers = make([]layers.ActivationLayer, m.depth)
	m.normalizationLayers = make([]layers.NormalizationLayer, m.depth)

	initActivationLayer := layers.InitSigmoidLayer
	switch a {
	case ActivationAlgorismSigmoid:
		initActivationLayer = layers.InitSigmoidLayer
	case ActivationAlgorismReLu:
		initActivationLayer = layers.InitReLuLayer
	default:
		initActivationLayer = layers.InitSigmoidLayer
	}

	initNormalizationLayer := layers.InitNoNormalizationLayer
	switch n {
	case NormalizationAlgorismBatchNorm:
		initNormalizationLayer = layers.InitBatchNormLayer
	case NormalizationAlgorismNo:
		initNormalizationLayer = layers.InitNoNormalizationLayer
	}

	for d := 0; d < depth; d++ {
		w := makeRandSliceFloat64(neurons[d]*neurons[d+1], weightInitStd)
		b := makeRandSliceFloat64(neurons[d+1], weightInitStd)

		weight := mat.NewDense(neurons[d], neurons[d+1], w)
		bias := mat.NewDense(1, neurons[d+1], b)

		m.params.Weight[d] = weight
		m.params.Bias[d] = bias
		m.params.Beta[d] = makeSliceFloat64(neurons[d+1], 0.0)
		m.params.Gamma[d] = makeSliceFloat64(neurons[d+1], 1.0)

		m.affineLayers[d] = layers.InitAffineLayer(weight, bias)
		m.normalizationLayers[d] = initNormalizationLayer(m.params.Beta[d], m.params.Gamma[d])

		if d < depth-1 {
			m.activationLayers[d] = initActivationLayer()
		} else {
			m.activationLayers[d] = layers.InitIdentityLayer()
		}
	}

	m.lastLayer = layers.InitSoftmaxWithLossLayer()

	return &m, nil
}

func (m *MultiLayerNet) Predict(x *mat.Dense) *mat.Dense {
	for i := 0; i < m.depth; i++ {
		x = m.affineLayers[i].Forward(x)
		x = m.activationLayers[i].Forward(x)
		x = m.normalizationLayers[i].Forward(x)
	}

	return x
}

func (m *MultiLayerNet) Loss(x, t *mat.Dense) float64 {
	y := m.Predict(x)
	return m.lastLayer.Forward(y, t)
}

func (m *MultiLayerNet) Accuracy(x, t *mat.Dense) float64 {
	batchSize, _ := x.Dims()

	y := m.Predict(x)
	sum := 0.0

	for i := 0; i < batchSize; i++ {
		if t.At(i, argmaxOnVec(y.RowView(i))) == 1.0 {
			sum = sum + 1.0
		}
	}

	return sum / float64(batchSize)
}

func (m *MultiLayerNet) NumericalGradient(x, t *mat.Dense) *Params {
	f := func(w *mat.Dense) float64 {
		return m.Loss(x, t)
	}

	grads := Params{
		Weight: make([]*mat.Dense, m.depth),
		Bias:   make([]*mat.Dense, m.depth),
		Depth:  m.depth,
	}

	for d := 0; d < m.depth; d++ {
		grads.Weight[d] = numericalGradient(f, m.params.Weight[d])
		grads.Bias[d] = numericalGradient(f, m.params.Bias[d])
	}

	return &grads
}

func (m *MultiLayerNet) Gradient(x, t *mat.Dense) *Params {
	m.Loss(x, t)

	dout := m.lastLayer.Backward(1.0)

	for i := m.depth - 1; i >= 0; i-- {
		dout = m.normalizationLayers[i].Backward(dout)
		dout = m.activationLayers[i].Backward(dout)
		dout = m.affineLayers[i].Backward(dout)
	}

	weight := make([]*mat.Dense, m.depth)
	bias := make([]*mat.Dense, m.depth)
	for i := 0; i < m.depth; i++ {
		weight[i] = m.affineLayers[i].GetDW()
		bias[i] = m.affineLayers[i].GetDB()
	}

	grads := Params{
		Weight: weight,
		Bias:   bias,
	}

	return &grads
}

func (m *MultiLayerNet) GetParams() *Params {
	return m.params
}

func (m *MultiLayerNet) GetDepth() int {
	return m.depth
}
