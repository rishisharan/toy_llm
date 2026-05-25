package main

type FeedForward struct {
	W1, W2 Matrix
	B1, B2 []float32
}

func NewFeedFroward(embedDim, hiddenDim int) FeedForward {

	w1 := newMatrix(embedDim, hiddenDim)
	w2 := newMatrix(hiddenDim, embedDim)
	for i := range w1.Data {
		w1.Data[i] = (randFloat() - 0.5) * 0.1
	}

	for i:= range w2.Data {
		w2.Data[i] = (randFloat() - 0.5) * 0.1
	}

	return FeedForward{
		W1: w1, W2: w2,
		B1: make([]float32, hiddenDim),
		B2: make([]float32, embedDim),
	}
}

func relu(x float32) float32 {
    if x > 0 { return x }
    return 0
}

func (ff FeedForward) Forward(x Matrix) Matrix {
    // step 1 — expand
    hidden := MatMul(x, ff.W1)
    for r := 0; r < hidden.Rows; r++ {
        for c := 0; c < hidden.Cols; c++ {
            v := hidden.Get(r, c) + ff.B1[c]
            hidden.Set(r, c, relu(v))
        }
    }
    // step 2 — compress
    out := MatMul(hidden, ff.W2)
    for r := 0; r < out.Rows; r++ {
        for c := 0; c < out.Cols; c++ {
            out.Set(r, c, out.Get(r, c)+ff.B2[c])
        }
    }
    return out
}