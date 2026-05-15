package main
import "math/rand"
import "fmt"
import "time"

func main() {
	rand.Seed(time.Now().UnixNano())
	fmt.Println("Welcome to Toy LLM project.")
	fmt.Println("Step 1: Tokenize.")
	word := "Hello"
	encodedMap, decodedMap := tokenize(word)
	fmt.Println("Encoded:", encodedMap)
	fmt.Println("Decoded:", decodedMap)
	fmt.Println("Step 1: Tokenization complete.")

	fmt.Println("Step 2: Embeddings.")
	a := newMatrix(3, 4)
	a.Set(0, 0, 1)
	a.Set(0, 1, 2)
	a.Set(0, 2, 3)
	a.Set(0, 3, 4)

	a.Set(1, 0, 4)
	a.Set(1, 1, 5)
	a.Set(1, 2, 6)
	a.Set(1, 3, 7)

	a.Set(2, 0, 4)
	a.Set(2, 1, 5)
	a.Set(2, 2, 6)
	a.Set(2, 3, 7)

	for i := range a.Data {
		a.Data[i] = float32(i) * 0.1
	}

	out := attention(a, a, a)
	fmt.Println("Attention output ", out.Rows, out.Cols)

	// use your encoded map from step 1
	tok := NewEmbedding(4, 8)   // 4 unique chars, 8 dimensions
	ids := []int{0, 1, 2, 2, 3} // "Hello" encoded

	outt := tok.Forward(ids)
	fmt.Println("Embedding output:", outt.Rows, outt.Cols) // should print 5 8
	fmt.Println("Embedding output:", outt.Data)
}
