package main
import "math/rand"

type Matrix struct {
	Data       []float32
	Rows, Cols int
}

func newMatrix(rows, cols int) Matrix {
	// ro := []float32{}
	return Matrix{
		Data: make([]float32, rows*cols),
		Rows: rows,
		Cols: cols,
	}
}

func (m Matrix) Get(rows, cols int) float32 {
	position := rows*m.Cols + cols
	return m.Data[position]
}

func (m Matrix) Set(rows, cols int, value float32) {
	position := rows*m.Cols + cols
	m.Data[position] = value
}

func MatMul(a, b Matrix) Matrix {
	result := newMatrix(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			sum := float32(0)
			for k := 0; k < a.Cols; k++ {
				// multiply and accumulate
				sum += a.Get(i, k) * b.Get(k, j)
			}
			result.Set(i, j, sum)
		}
	}
	return result
}

func (m Matrix) applyTranspose(a Matrix) Matrix {
	result := newMatrix(a.Cols, a.Rows)
	for r := 0; r < a.Rows; r++ {
		for c := 0; c < a.Cols; c++ {
			// one line here
			result.Set(c, r, a.Get(r, c))
		}
	}
	return result
}


func randFloat() float32 {
    return float32(rand.Float64())
}