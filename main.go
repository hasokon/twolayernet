package main

import (
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/hasokon/mnist"
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
	hs := 100
	os := 10
	bs := 100
	loop := 10000

	trainimages, trainlabels, testimages, testlabels := mnist.GetDataForNN()

	tri := mat.NewDense(mnist.TrainDataSize, is, trainimages)
	trl := mat.NewDense(mnist.TrainDataSize, os, trainlabels)
	tei := mat.NewDense(mnist.TestDataSize, is, testimages)
	tel := mat.NewDense(mnist.TestDataSize, os, testlabels)

	net := InitNet(is, hs, os, 0.01, 0.01)

	for i := 0; i < loop; i++ {
		batchData, batchLabel := GetBatchData(tri, trl, bs)
		grads := net.Gradient(batchData, batchLabel)
		net.ParamUpdate(grads)

		if i%1000 == 0 {
			testData, testLabel := GetBatchData(tei, tel, bs)
			fmt.Printf("%6d: Loss=%f, Accuracy=%3.1f%%\n", i, net.Loss(batchData, batchLabel), net.Accuracy(testData, testLabel)*100)
		}
	}
}

// package main

// import (
// 	"image"
// 	"image/color"
// 	"math/rand"
// 	"time"

// 	"github.com/hajimehoshi/ebiten"
// 	"github.com/hasokon/mnist"
// )

// const (
// 	width     = 640
// 	height    = 480
// 	intervalH = 3
// 	intervalW = 2
// )

// func GetUpdate() func(*ebiten.Image) error {
// 	mnist, _ := mnist.InitMNIST()

// 	// imgwidth := mnist.TrainImageWidth
// 	// imgheight := mnist.TrainImageHeight
// 	// numOfImages := mnist.TrainDataSize
// 	// images := mnist.TrainImages

// 	imgwidth := mnist.TestImageWidth
// 	imgheight := mnist.TestImageHeight
// 	numOfImages := mnist.TestDataSize
// 	images := mnist.TestImages

// 	widthNum := width / (imgwidth + intervalW*2)
// 	heightNum := height / (imgheight + intervalH*2)

// 	displayImages := make([][]*ebiten.Image, heightNum)

// 	rand.Seed(time.Now().UnixNano())
// 	index := rand.Perm(numOfImages)

// 	for i := 0; i < heightNum; i++ {
// 		displayImages[i] = make([]*ebiten.Image, widthNum)
// 		for j := 0; j < widthNum; j++ {
// 			displayImages[i][j], _ = ebiten.NewImageFromImage(
// 				&image.Gray{
// 					Pix:    images[index[i*widthNum+j]],
// 					Stride: imgwidth,
// 					Rect:   image.Rect(0, 0, imgwidth, imgheight),
// 				}, ebiten.FilterLinear)
// 		}
// 	}

// 	return func(screen *ebiten.Image) error {

// 		screen.Clear()
// 		screen.Fill(color.White)

// 		op := &ebiten.DrawImageOptions{}
// 		op.GeoM.Translate(float64(intervalW), float64(intervalH))

// 		for i := 0; i < heightNum; i++ {
// 			for j := 0; j < widthNum; j++ {
// 				screen.DrawImage(displayImages[i][j], op)
// 				op.GeoM.Translate(float64(intervalW*2+imgwidth), 0)
// 			}
// 			op.GeoM.Translate(float64((intervalW*2+imgwidth)*widthNum*-1), float64(intervalH*2+imgheight))
// 		}
// 		return nil
// 	}
// }

// func main() {
// 	ebiten.Run(GetUpdate(), width, height, 2.0, "MNIST Images")
// }
