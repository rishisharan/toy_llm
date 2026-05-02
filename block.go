// block.go
package main

import "math"

type TransformerBlock struct {
	FF             *FeedForward
	lastInput      Matrix
	lastInputGrad  Matrix
	lastAttnScores Matrix
	lastAttnOut    Matrix
}

func NewTransformerBlock(embedDim, hiddenDim int) TransformerBlock {
	ff := NewFeedForward(embedDim, hiddenDim)
	return TransformerBlock{
		FF: &ff,
	}
}

func (b *TransformerBlock) Forward(x Matrix) Matrix {
	b.lastInput = x
	// compute and cache attention scores
	scores := MatMul(x, Transpose(x))
	scale := float32(math.Sqrt(float64(x.Cols)))
	for i := range scores.Data {
		scores.Data[i] /= scale
	}
	for r := 0; r < scores.Rows; r++ {
		row := scores.Data[r*scores.Cols : (r+1)*scores.Cols]
		copy(row, softMax(row))
	}
	b.lastAttnScores = scores

	attnOut := MatMul(scores, x)
	b.lastAttnOut = attnOut

	// residual
	for i := range x.Data {
		attnOut.Data[i] += x.Data[i]
	}

	ffOut := b.FF.Forward(attnOut)
	for i := range ffOut.Data {
		ffOut.Data[i] += attnOut.Data[i]
	}
	return ffOut
}

// func (b *TransformerBlock) Backward(dOut Matrix) {
// 	dFF := b.FF.Backward(dOut)
// 	for i := range dFF.Data {
// 		dFF.Data[i] += dOut.Data[i]
// 	}
// 	b.lastInputGrad = dFF
// }

func (b *TransformerBlock) Backward(dOut Matrix) {
	// residual through feed-forward
	dFF := b.FF.Backward(dOut)
	for i := range dFF.Data {
		dFF.Data[i] += dOut.Data[i]
	}

	// attention backward
	dQ, dK, _ := attentionBackward(
		b.lastInput, b.lastInput, b.lastInput,
		b.lastAttnScores, b.lastAttnOut, dFF,
	)

	// combine Q and K gradients — both came from same input x
	dInput := NewMatrix(b.lastInput.Rows, b.lastInput.Cols)
	for i := range dInput.Data {
		dInput.Data[i] = dQ.Data[i] + dK.Data[i] + dFF.Data[i]
	}

	b.lastInputGrad = dInput
}

func attentionBackward(Q, K, V, scores, out, dOut Matrix) (Matrix, Matrix, Matrix) {
	// gradient through V
	dV := MatMul(Transpose(scores), dOut)

	// gradient through scores
	dScores := MatMul(dOut, Transpose(V))

	// gradient through softmax
	for r := 0; r < dScores.Rows; r++ {
		row := scores.Data[r*scores.Cols : (r+1)*scores.Cols]
		dRow := dScores.Data[r*dScores.Cols : (r+1)*dScores.Cols]
		sum := float32(0)
		for i, p := range row {
			sum += p * dRow[i]
		}
		for i, p := range row {
			dScores.Data[r*dScores.Cols+i] = p * (dRow[i] - sum)
		}
	}

	// gradient through scale
	scale := float32(math.Sqrt(float64(Q.Cols)))
	for i := range dScores.Data {
		dScores.Data[i] /= scale
	}

	// gradient through Q and K
	dQ := MatMul(dScores, K)
	dK := MatMul(Transpose(dScores), Q)

	return dQ, dK, dV
}
