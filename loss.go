package main
import "math"

func crossEntropyLoss(logits Matrix, targets []int) float32 {
	totalLoss := float32(0)
    
    for r := 0; r < logits.Rows; r++ {
        // get row r from logits
        row := logits.Data[r*logits.Cols : (r+1)*logits.Cols]
        
        // YOUR TURN — write these three lines:
        // 1. get probs by calling softmax on row
		probs := softmax(row)
        
		// 2. get correctProb using targets[r] as index
        
		// 3. add -log(correctProb) to totalLoss
		correctProb := probs[targets[r]]         // line 2
		totalLoss += float32(-math.Log(float64(correctProb) + 1e-9)) 
    }
    
    return totalLoss / float32(logits.Rows)
}