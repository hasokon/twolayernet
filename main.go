package main

import (
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func main() {
	is := 784
	hs := 50
	os := 10
	bs := 100
	loop := 10000

	net := InitNet(is, hs, os, 0.1, 0.01)

	rand.Seed(time.Now().UnixNano())
	input := mat.NewDense(bs, is, makeRandSliceInt(bs*is))
	t := mat.NewDense(bs, os, nil)
	for i := 0; i < bs; i++ {
		truearg := rand.Intn(os)
		t.Set(i, truearg, 1.0)
	}

	fmt.Printf("first:")
	fmt.Println(net.Loss(input, t))

	for i := 0; i < loop; i++ {
		grads := net.Gradient(input, t)
		net.ParamUpdate(grads)
		if i%1000 == 0 {
			fmt.Printf("%d:", i+1)
			fmt.Println(net.Loss(input, t))
		}
	}
}
