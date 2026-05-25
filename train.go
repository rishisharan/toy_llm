package main

import (
	"fmt"
)

func train(model *Model, ids []int, steps int, lr float32) {
	targets := make([]int, len(ids))
	for i := 0; i < len(ids)-1; i++ {
		targets[i] = ids[i+1]
	}
	targets[len(ids)-1] = ids[0]

	for step := 0; step < steps; step++ {
		logits := forwardPass(model, ids)
		loss := crossEntropyLoss(logits, targets)

		for i := range model.Embedding.Weight.Data {
			numericalGradient(model, ids, targets, &model.Embedding.Weight.Data[i], lr)
		}
		for i := range model.FF.W1.Data {
			numericalGradient(model, ids, targets, &model.FF.W1.Data[i], lr)
		}
		for i := range model.FF.W2.Data {
			numericalGradient(model, ids, targets, &model.FF.W2.Data[i], lr)
		}
		for i := range model.ProjW.Data {
			numericalGradient(model, ids, targets, &model.ProjW.Data[i], lr)
		}

		if step%50 == 0 {
			fmt.Printf("step %d loss %.4f\n", step, loss)
		}
	}
}

func numericalGradient(model *Model, ids []int, targets []int, param *float32, lr float32) {
    h := float32(0.001)

    original := *param

    *param += h
    logits := forwardPass(model, ids)
    lossPlus := crossEntropyLoss(logits, targets)

    *param -= 2 * h
    logits = forwardPass(model, ids)
    lossMinus := crossEntropyLoss(logits, targets)

    *param = original
    grad := (lossPlus - lossMinus) / (2 * h)
    
    // debug

    
    *param -= lr * grad
}

func forwardPass(model *Model, ids []int) Matrix {
	x := model.Embedding.Forward(ids)
	x = addPositionalEncoding(x)
	x = attention(x, x, x)
	x = model.FF.Forward(x)
	logits := MatMul(x, model.ProjW)
	return logits
}
