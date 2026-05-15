package main
import "math"
func attention(Q, K, V Matrix) Matrix {
	scores := MatMul(Q, Q.applyTranspose(K))
	scale := float32(math.Sqrt(float64(Q.Cols)))
	for i := range scores.Data {
		scores.Data[i] /= scale
	}

	for r := 0; r < scores.Rows; r++ {
		row := scores.Data[r*scores.Cols : (r+1)*scores.Cols]
		copy(row, softmax(row))
	}

	return MatMul(scores, V)
}