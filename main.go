package main

import (
	"fmt"
	"math/rand"
	"time"
)
func main() {
    rand.Seed(time.Now().UnixNano())
    fmt.Println("Welcome to Toy LLM project.")

    // Step 1: Tokenize
    fmt.Println("\nStep 1: Tokenize.")
    word := "Hello"
    encodedMap, decodedMap := tokenize(word)
    fmt.Println("Encoded:", encodedMap)
    fmt.Println("Decoded:", decodedMap)

    // Step 2: Setup
    ids := []int{0, 1, 2, 2, 3} // "Hello" encoded
    model := NewModel(4, 8, 32)  // vocabSize=4, embedDim=8, hiddenDim=32

    // Check initial loss
    initialLogits := forwardPass(&model, ids)
    initialLoss := crossEntropyLoss(initialLogits, []int{1, 2, 2, 3, 0})
    fmt.Printf("\nInitial loss: %.4f\n", initialLoss)

    // Step 9: Train
    fmt.Println("\nStep 9: Training...")
    train(&model, ids, 2500, 0.1)

	// Step 10: Generate
	fmt.Println("\nStep 10: Generating...")
	result := generate(&model, []int{0}, 10, 0.1)
	for _, id := range result {
		fmt.Print(decodedMap[id])
	}
	fmt.Println()
}