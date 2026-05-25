package main

func generate(model *Model, ids []int, maxTokens int, temperature float32) []int {
    for i := 0; i < maxTokens; i++ {
        logits := forwardPass(model, ids)
        
        // take only last row — prediction for next token
        lastRow := logits.Data[(logits.Rows-1)*logits.Cols:]
        
        // apply temperature
        scaled := make([]float32, len(lastRow))
        for j, v := range lastRow {
            scaled[j] = v / temperature
        }
        
        // get probabilities
        probs := softmax(scaled)
        
        // pick next token
        next := argmax(probs)
        ids = append(ids, next)
    }
    return ids
}

func argmax(probs []float32) int {
    maxIdx := 0
    for i, v := range probs {
        if v > probs[maxIdx] {
            maxIdx = i
        }
    }
    return maxIdx
}