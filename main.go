package main

import (
	"fmt"
	"math"
)

type Matrix struct {
	Data       []float32
	Rows, Cols int
}

func NewMatrix(rows, cols int) Matrix {
	return Matrix{
		Data: make([]float32, rows*cols),
		Rows: rows,
		Cols: cols,
	}
}

func (m Matrix) Get(r, c int) float32 {
	return m.Data[r*m.Cols+c]
}

func (m Matrix) Set(r, c int, v float32) {
	m.Data[r*m.Cols+c] = v
}

func MatMul(a, b Matrix) Matrix {
	result := NewMatrix(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ { // each row of A
		for j := 0; j < b.Cols; j++ { // each col of B
			sum := float32(0)
			for k := 0; k < a.Cols; k++ { // walk across & down
				sum += a.Get(i, k) * b.Get(k, j)
			}
			result.Set(i, j, sum)
		}
	}
	return result
}

func main() {
	text := "hello world how are you doing today"

	fmt.Println("=== Step 1: Tokenizing ===")
	tok := NewTokenizer(text)
	fmt.Printf("vocab size: %d unique characters\n", tok.Size)
	fmt.Printf("encoded: %v\n", tok.Encode("hello"))

	fmt.Println("\n=== Step 2: Building model ===")
	model := NewModel(tok.Size, 16, 64)
	fmt.Println("transformer ready")

	fmt.Println("\n=== Step 3: Training ===")
	TrainBackprop(&model, text, tok, 500, 0.001)

	fmt.Println("\n=== Step 4: Generating ===")
	fmt.Println(Generate(&model, tok, "h", 50, 0.3))

}

func softMax(x []float32) []float32 {
	out := make([]float32, len(x))
	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range x {
		out[i] = float32(math.Exp(float64(v - max)))
		sum += out[i]
	}

	for i := range out {
		out[i] /= sum
	}
	return out
}

// Transpose flips a matrix — rows become cols, cols become rows
// (2x3) → (3x2)
func Transpose(a Matrix) Matrix {
	result := NewMatrix(a.Cols, a.Rows)
	for r := 0; r < a.Rows; r++ {
		for c := 0; c < a.Cols; c++ {
			result.Set(c, r, a.Get(r, c))
		}
	}
	return result
}

func Attention(Q, K, V Matrix) Matrix {
	// Step 1 — score every token against every other token
	// scores[i][j] = "how much should token i attend to token j?"
	scores := MatMul(Q, Transpose(K))

	// Step 2 — scale down to stop values exploding
	scale := float32(math.Sqrt(float64(Q.Cols)))
	for i := range scores.Data {
		scores.Data[i] /= scale
	}

	// Step 3 — softmax each row so scores become probabilities
	for r := 0; r < scores.Rows; r++ {
		row := scores.Data[r*scores.Cols : (r+1)*scores.Cols]
		softRow := softMax(row)
		copy(row, softRow)
	}

	// Step 4 — weighted sum of values
	return MatMul(scores, V)
}
