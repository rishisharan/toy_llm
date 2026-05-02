// loss.go
package main

import "math"

// targets[i] = the correct next token id at position i
func CrossEntropyLoss(logits Matrix, targets []int) float32 {
	totalLoss := float32(0)
	for r := 0; r < logits.Rows; r++ {
		// get the row of logits for this position
		row := logits.Data[r*logits.Cols : (r+1)*logits.Cols]

		// convert logits to probabilities
		probs := softMax(row)

		// loss = -log(probability of correct token)
		correctProb := probs[targets[r]]
		totalLoss += float32(-math.Log(float64(correctProb) + 1e-9))
	}
	return totalLoss / float32(logits.Rows)
}

func CrossEntropyBackward(logits Matrix, targets []int) Matrix {
    grad := NewMatrix(logits.Rows, logits.Cols)
    for r := 0; r < logits.Rows; r++ {
        row  := logits.Data[r*logits.Cols : (r+1)*logits.Cols]
        probs := softMax(row)
        for c := 0; c < logits.Cols; c++ {
            grad.Set(r, c, probs[c])
        }
        // subtract 1 from the correct token
        grad.Data[r*logits.Cols+targets[r]] -= 1
    }
    // average over sequence length
    for i := range grad.Data {
        grad.Data[i] /= float32(logits.Rows)
    }
    return grad
}