package layers

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type BatchNormLayer struct {
	gamma []float64
	beta  []float64
	diff  *mat.Dense
	s2b   []float64
	den   []float64
	norm  *mat.Dense

	dgamma []float64
	dbeta  []float64
}

func InitBatchNormLayer(g, b []float64) NormalizationLayer {
	return &BatchNormLayer{
		gamma: g,
		beta:  b,
	}
}

// func (b *BatchNormLayer) Forward(x *mat.Dense) *mat.Dense {
// 	return x
// }

// func (b *BatchNormLayer) Backward(dout *mat.Dense) *mat.Dense {
// 	return dout
// }

func (b *BatchNormLayer) Forward(x *mat.Dense) *mat.Dense {
	r, c := x.Dims()
	out := mat.NewDense(r, c, nil)
	b.diff = mat.NewDense(r, c, nil)
	mb := make([]float64, c)
	b.s2b = make([]float64, c)
	b.den = make([]float64, c)
	b.norm = mat.NewDense(r, c, nil)

	// Calc mybeta
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			mb[j] = mb[j] + x.At(i, j)
		}
	}

	for i := 0; i < c; i++ {
		mb[i] = mb[i] / float64(r)
	}

	// Calc sigmma**2 beta
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			tmp := x.At(i, j) - mb[j]
			b.diff.Set(i, j, tmp)
			b.s2b[j] = b.s2b[j] + tmp*tmp
		}
	}

	epsilon := math.Pow10(-7)
	for i := 0; i < c; i++ {
		b.s2b[i] = b.s2b[i] / float64(r)
		b.den[i] = 1.0 / math.Sqrt(b.s2b[i]+epsilon)
		b.s2b[i] = 1.0 / (b.s2b[i] + epsilon)
	}

	// Normalization and Scale-Shift
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			tmp := b.diff.At(i, j) * b.den[j]
			b.norm.Set(i, j, tmp)
			out.Set(i, j, tmp*b.gamma[j]+b.beta[j])
		}
	}

	return out
}

func (b *BatchNormLayer) Backward(dout *mat.Dense) *mat.Dense {
	r, c := dout.Dims()
	dx := mat.NewDense(r, c, nil)
	b.dbeta = make([]float64, c)
	b.dgamma = make([]float64, c)
	tmp0 := mat.NewDense(r, c, nil)
	tmp1 := make([]float64, c)
	sums := make([]float64, c)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			b.dbeta[j] = b.dbeta[j] + dout.At(i, j)
			b.dgamma[j] = b.dgamma[j] + b.norm.At(i, j)*dout.At(i, j)
			tmp0.Set(i, j, b.gamma[j]*dout.At(i, j))
			tmp1[j] = tmp1[j] + tmp0.At(i, j)*b.diff.At(i, j)
		}
	}

	for i := 0; i < c; i++ {
		tmp1[i] = tmp1[i] / float64(r)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum := b.den[j] * (tmp0.At(i, j) - b.diff.At(i, j)*b.s2b[j]*tmp1[j])
			sums[j] = sums[j] + sum
			dx.Set(i, j, sum)
		}
	}

	for i := 0; i < c; i++ {
		sums[i] = sums[i] / float64(r)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			dx.Set(i, j, dx.At(i, j)-sums[j])
		}
	}

	return dx
}
