// feedforward.go
package main

type FeedForward struct {
	W1, W2         Matrix    // two weight matrices
	B1, B2         []float32 // biases
	W1Grad, W2Grad Matrix
	B1Grad, B2Grad []float32
	// cache
	lastInput  Matrix
	lastHidden Matrix
}

func NewFeedForward(embedDim, hiddenDim int) FeedForward {
	w1 := NewMatrix(embedDim, hiddenDim)
	w2 := NewMatrix(hiddenDim, embedDim)
	for i := range w1.Data {
		w1.Data[i] = (randFloat() - 0.5) * 0.01
	}
	for i := range w2.Data {
		w2.Data[i] = (randFloat() - 0.5) * 0.01
	}
	return FeedForward{
		W1: w1, W2: w2,
		B1:     make([]float32, hiddenDim),
		B2:     make([]float32, embedDim),
		W1Grad: NewMatrix(embedDim, hiddenDim),
		W2Grad: NewMatrix(hiddenDim, embedDim),
		B1Grad: make([]float32, hiddenDim),
		B2Grad: make([]float32, embedDim),
	}
}

func relu(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0
}

func (ff *FeedForward) Forward(x Matrix) Matrix {
	ff.lastInput = x
	hidden := MatMul(x, ff.W1)
	for r := 0; r < hidden.Rows; r++ {
		for c := 0; c < hidden.Cols; c++ {
			v := hidden.Get(r, c) + ff.B1[c]
			hidden.Set(r, c, relu(v))
		}
	}
	ff.lastHidden = hidden
	out := MatMul(hidden, ff.W2)
	for r := 0; r < out.Rows; r++ {
		for c := 0; c < out.Cols; c++ {
			out.Set(r, c, out.Get(r, c)+ff.B2[c])
		}
	}
	return out
}

func (ff *FeedForward) Backward(dOut Matrix) Matrix {
	// gradient through B2
	for c := 0; c < dOut.Cols; c++ {
		for r := 0; r < dOut.Rows; r++ {
			ff.B2Grad[c] += dOut.Get(r, c)
		}
	}
	// gradient through W2
	ff.W2Grad = MatMul(Transpose(ff.lastHidden), dOut)

	// gradient back through hidden
	dHidden := MatMul(dOut, Transpose(ff.W2))

	// gradient through ReLU
	for i, v := range ff.lastHidden.Data {
		if v <= 0 {
			dHidden.Data[i] = 0
		}
	}

	// gradient through B1
	for c := 0; c < dHidden.Cols; c++ {
		for r := 0; r < dHidden.Rows; r++ {
			ff.B1Grad[c] += dHidden.Get(r, c)
		}
	}

	// gradient through W1
	ff.W1Grad = MatMul(Transpose(ff.lastInput), dHidden)

	// gradient to pass back
	return MatMul(dHidden, Transpose(ff.W1))
}
