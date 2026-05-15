package main

import "fmt"

func main() {
	fmt.Println("Welcome to Toy LLM project.")
	fmt.Println("Step 1: Tokenize.")
	word := "Hello"
	encodedMap, decodedMap := tokenize(word)
	fmt.Println("Encoded:", encodedMap)
	fmt.Println("Decoded:", decodedMap)
	fmt.Println("Step 1: Tokenization complete.")

	fmt.Println("Step 2: Embeddings.")
	embedding(word)
	a := newMatrix(2, 3)
	a.Set(0, 0, 1)
	a.Set(0, 1, 2)
	a.Set(0, 2, 3)
	a.Set(1, 0, 4)
	a.Set(1, 1, 5)
	a.Set(1, 2, 6)

	b := newMatrix(3, 2)
	b.Set(0, 0, 7)
	b.Set(0, 1, 8)
	b.Set(1, 0, 9)
	b.Set(1, 1, 10)
	b.Set(2, 0, 11)
	b.Set(2, 1, 12)

	c := MatMul(a, b)
	fmt.Println(c.Data) // expect [58 64 139 154]

	soft := softmax(c.Data)
	c.Data = soft
	fmt.Println("Softmax", soft)
	fmt.Println("Softmax matrix", c)
	fmt.Println("Transpose", c.applyTranspose(c))

}
