package neuralnetwork

import "gonum.org/v1/gonum/mat"

type NeuralNetwork interface {
	Predict(input *mat.Dense) *mat.Dense
	Loss(x, t *mat.Dense) float64
	Accuracy(x, t *mat.Dense) float64
	NumericalGradient(x, t *mat.Dense) *Params
	Gradient(x, t *mat.Dense) *Params
	GetParams() *Params
}
