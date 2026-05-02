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
	// 2x3 matrix
	a := NewMatrix(2, 3)
	a.Set(0, 0, 1)
	a.Set(0, 1, 2)
	a.Set(0, 2, 3)
	a.Set(1, 0, 4)
	a.Set(1, 1, 5)
	a.Set(1, 2, 6)

	// 3x2 matrix
	b := NewMatrix(3, 2)
	b.Set(0, 0, 7)
	b.Set(0, 1, 8)
	b.Set(1, 0, 9)
	b.Set(1, 1, 10)
	b.Set(2, 0, 11)
	b.Set(2, 1, 12)

	c := MatMul(a, b)
	fmt.Println(c.Data) // expect [58 64 139 154]

	scores := []float32{2.0, 1.0, 0.5}
	fmt.Println(softMax(scores))

	t := Transpose(a)
	fmt.Println(t.Rows, t.Cols)

	// in main()
	Q := NewMatrix(3, 4) // 3 tokens, 4 dimensional
	K := NewMatrix(3, 4)
	V := NewMatrix(3, 4)

	// fill with some dummy values
	for i := range Q.Data {
		Q.Data[i] = float32(i) * 0.1
	}
	for i := range K.Data {
		K.Data[i] = float32(i) * 0.1
	}
	for i := range V.Data {
		V.Data[i] = float32(i) * 0.1
	}

	out := Attention(Q, K, V)
	fmt.Println(out.Rows, out.Cols) // should print: 3 4

	tok := NewTokenizer("hello world")
	ids := tok.Encode("hello")

	emb := NewEmbedding(tok.Size, 8) // 8-dimensional embeddings
	ouut := emb.Forward(ids)
	ouut = AddPosEncoding(ouut)

	fmt.Println(ouut.Rows, ouut.Cols) // 5 8 — 5 tokens, 8 dims each

	fmt.Println(ids)             // some numbers
	fmt.Println(tok.Decode(ids)) // "hello"
	fmt.Println("vocab size:", tok.Size)

	ff := NewFeedForward(8, 32)   // embedDim=8, hiddenDim=32
	ot := ff.Forward(ouut)        // x is your 5x8 matrix from before
	fmt.Println(ot.Rows, ot.Cols) // 5 8 — same shape in, same shape out

	block := NewTransformerBlock(8, 32)
	ouuut := block.Forward(ouut)
	fmt.Println(ouuut.Rows, ouuut.Cols)

	model := NewModel(tok.Size, 8, 32)

	logits := model.Forward(ids)

	fmt.Println(logits.Rows, logits.Cols) // 5 8 — 5 tokens, 8 vocab scores each

	// targets = next token at each position
	// "h"→"e"→"l"→"l"→"o"
	targets := ids[1:]
	targets = append(targets, ids[0]) // wrap around for last position

	loss := CrossEntropyLoss(logits, targets)
	fmt.Printf("initial loss: %.4f\n", loss)

	// Train(&model, ids, 500, 0.01)

	// fmt.Println(Generate(&model, tok, "h", 20, 0.6)) // lower temperature = more confident
	// fmt.Println(Generate(&model, tok, "h", 20, 0.1))

	// data, _ := os.ReadFile("transcript.txt")
	// text := string(data)
	text := "hello world how are you doing today hello world how are you"
	// toke := NewTokenizer(text)
	// modele := NewModel(toke.Size, 8, 16) // bigger dims for real data
	// idse := toke.Encode(text[:100])      // start with first 500 chars

	// Train(&modele, idse, 50, 0.01)

	// fmt.Println(Generate(&modele, toke, "T", 50, 0.6))

	toke := NewTokenizer(text)
	modele := NewModel(toke.Size, 32, 128)

	TrainBackprop(&modele, text, toke, 6000, 0.001)
	fmt.Println(Generate(&modele, toke, "hello world", 100, 0.6))


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
