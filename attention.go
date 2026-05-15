package main
import "math"
func attention(Q, K, V Matrix) Matrix {
	scores := MatMul(Q, Q.applyTranspose(K)) //Q × K^T  → how much does each token care about every other token?
	scale := float32(math.Sqrt(float64(Q.Cols))) //÷ scale  → prevent numbers getting too large
	for i := range scores.Data {  
		scores.Data[i] /= scale
	}

	for r := 0; r < scores.Rows; r++ {
		row := scores.Data[r*scores.Cols : (r+1)*scores.Cols]
		copy(row, softmax(row))
	}

	return MatMul(scores, V)
}