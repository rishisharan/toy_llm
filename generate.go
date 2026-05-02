// generate.go
package main

import (
    "math/rand"
    
)

func Generate(model *Model, tok Tokenizer, prompt string, maxTokens int, temperature float32) string {
    ids := tok.Encode(prompt)

    for i := 0; i < maxTokens; i++ {
        // forward pass — get logits for current sequence
        logits := model.Forward(ids)

        // take only the last row — prediction for next token
        lastRow := logits.Data[(logits.Rows-1)*logits.Cols:]

        // apply temperature — higher = more random, lower = more confident
        scaled := make([]float32, len(lastRow))
        for j, v := range lastRow {
            scaled[j] = v / temperature
        }

        // sample from the distribution
        probs := softMax(scaled)
        next := sample(probs)
        ids = append(ids, next)
    }

    return tok.Decode(ids)
}

func sample(probs []float32) int {
    r := rand.Float32()
    var cumulative float32
    for i, p := range probs {
        cumulative += p
        if r < cumulative {
            return i
        }
    }
    return len(probs) - 1
}