package main

import (
	"fmt"

	"github.com/hasokon/twolayernet/layers"
	"gonum.org/v1/gonum/mat"
)

func main() {

	relu := layers.InitSigmoidLayer()

	x := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	dout := mat.NewDense(2, 2, []float64{1, 2, 3, 4})

	fmt.Println(mat.Formatted(relu.Forward(x)))
	fmt.Println(mat.Formatted(relu.Backward(dout)))
	// is := 50
	// hs := 30
	// os := 10
	// bs := 100
	// loop := 1

	// net := InitNet(is, hs, os, 0.1, 0.01)

	// rand.Seed(time.Now().UnixNano())
	// input := mat.NewDense(bs, is, makeRandSliceInt(bs*is))
	// t := mat.NewDense(bs, os, nil)
	// for i := 0; i < bs; i++ {
	// 	truearg := rand.Intn(os)
	// 	t.Set(i, truearg, 1.0)
	// }

	// fmt.Printf("first:")
	// fmt.Println(net.Loss(input, t))

	// for i := 0; i < loop; i++ {
	// 	grads := net.NumericalGradient(input, t)
	// 	net.ParamUpdate(grads)
	// 	fmt.Printf("%d:", i+1)
	// 	fmt.Println(net.Loss(input, t))
	// }
}
