package main

import (
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/hasokon/mnist"
	"github.com/hasokon/twolayernet/neuralnetwork"
	"github.com/hasokon/twolayernet/optimizer"
)

func GetBatchData(d, l *mat.Dense, bs int) (bd, bl *mat.Dense) {
	len, datasize := d.Dims()
	_, labelsize := l.Dims()
	rand.Seed(time.Now().UnixNano())

	if len < bs {
		bs = len
	}

	bd = mat.NewDense(bs, datasize, nil)
	bl = mat.NewDense(bs, labelsize, nil)
	index := rand.Perm(len)

	for i := 0; i < bs; i++ {
		for j := 0; j < datasize; j++ {
			bd.Set(i, j, d.At(index[i], j))
		}
		for j := 0; j < labelsize; j++ {
			bl.Set(i, j, l.At(index[i], j))
		}
	}

	return
}

func main() {
	mnist, _ := mnist.InitMNIST()

	is := mnist.TrainImageWidth * mnist.TrainImageHeight
	os := 10
	neurons := []int{is, 500, os}
	bs := 100
	loop := 10000

	trainimages, trainlabels, testimages, testlabels := mnist.GetDataForNN()

	tri := mat.NewDense(mnist.TrainDataSize, is, trainimages)
	trl := mat.NewDense(mnist.TrainDataSize, os, trainlabels)
	tei := mat.NewDense(mnist.TestDataSize, is, testimages)
	tel := mat.NewDense(mnist.TestDataSize, os, testlabels)

	net, _ := neuralnetwork.InitMultiLayerNet(len(neurons)-1, neurons, 0.01, neuralnetwork.ActivationAlgorismReLu)
	opt := optimizer.InitOptimizer(net.GetDepth(), 0.001, optimizer.AlgorismSGD) //Sig=0.1, ReLu=0.001

	for i := 0; i < loop; i++ {
		batchData, batchLabel := GetBatchData(tri, trl, bs)
		if i%200 == 0 {
			testData, testLabel := GetBatchData(tei, tel, bs)
			fmt.Printf("%6d: Loss=%f, Accuracy=%3.1f%%\n", i, net.Loss(batchData, batchLabel), net.Accuracy(testData, testLabel)*100)
		}
		grads := net.Gradient(batchData, batchLabel)
		opt.Update(net.GetParams(), grads)
	}
}
