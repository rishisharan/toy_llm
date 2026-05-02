// train.go
package main

import "fmt"

func NumericalGradient(model *Model, ids []int, targets []int, param *float32) float32 {
	h := float32(0.001)

	*param += h
	x := model.Forward(ids)
	lossPlus := CrossEntropyLoss(x, targets)

	*param -= 2 * h
	x = model.Forward(ids)
	lossMinus := CrossEntropyLoss(x, targets)

	*param += h // restore
	return (lossPlus - lossMinus) / (2 * h)
}

func Train(model *Model, ids []int, steps int, lr float32) {
	for step := 0; step < steps; step++ {
		// targets = next token at each position
		targets := make([]int, len(ids))
		for i := 0; i < len(ids)-1; i++ {
			targets[i] = ids[i+1]
		}
		targets[len(ids)-1] = ids[0]

		// update every parameter
		allParams := collectParams(model)
		for _, p := range allParams {
			grad := NumericalGradient(model, ids, targets, p)
			*p -= lr * grad
		}

		if step%10 == 0 {
			logits := model.Forward(ids)
			loss := CrossEntropyLoss(logits, targets)
			fmt.Printf("step %d loss %.4f\n", step, loss)
		}
	}
}

func collectParams(model *Model) []*float32 {
	var params []*float32
	for i := range model.Embedding.Weight.Data {
		params = append(params, &model.Embedding.Weight.Data[i])
	}
	for i := range model.Block.FF.W1.Data {
		params = append(params, &model.Block.FF.W1.Data[i])
	}
	for i := range model.Block.FF.W2.Data {
		params = append(params, &model.Block.FF.W2.Data[i])
	}
	for i := range model.ProjW.Data {
		params = append(params, &model.ProjW.Data[i])
	}
	return params
}

func TrainBackprop(model *Model, text string, tok Tokenizer, steps int, lr float32) {
	ids := tok.Encode(text)
	targets := make([]int, len(ids))
	for i := 0; i < len(ids)-1; i++ {
		targets[i] = ids[i+1]
	}
	targets[len(ids)-1] = ids[0]

	for step := 0; step < steps; step++ {
		// forward
		logits := model.Forward(ids)
		loss := CrossEntropyLoss(logits, targets)

		// backward
		model.ZeroGrad()
		model.Backward(targets)
		clipGrads(model, 1.0)
		// update weights
		model.Step(lr)

		if step%50 == 0 {
			fmt.Printf("step %d loss %.4f\n", step, loss)
		}
	}
}

func (m *Model) ZeroGrad() {
	for i := range m.ProjWGrad.Data {
		m.ProjWGrad.Data[i] = 0
	}
	for i := range m.Block.FF.W1Grad.Data {
		m.Block.FF.W1Grad.Data[i] = 0
	}
	for i := range m.Block.FF.W2Grad.Data {
		m.Block.FF.W2Grad.Data[i] = 0
	}
	for i := range m.Embedding.WeightGrad.Data {
		m.Embedding.WeightGrad.Data[i] = 0
	}
	for i := range m.Block.FF.B1Grad {
		m.Block.FF.B1Grad[i] = 0
	}
	for i := range m.Block.FF.B2Grad {
		m.Block.FF.B2Grad[i] = 0
	}
}

func (m *Model) Step(lr float32) {
	for i := range m.ProjW.Data {
		m.ProjW.Data[i] -= lr * m.ProjWGrad.Data[i]
	}
	for i := range m.Block.FF.W1.Data {
		m.Block.FF.W1.Data[i] -= lr * m.Block.FF.W1Grad.Data[i]
	}
	for i := range m.Block.FF.W2.Data {
		m.Block.FF.W2.Data[i] -= lr * m.Block.FF.W2Grad.Data[i]
	}
	for i := range m.Embedding.Weight.Data {
		m.Embedding.Weight.Data[i] -= lr * m.Embedding.WeightGrad.Data[i]
	}
	for i := range m.Block.FF.B1 {
		m.Block.FF.B1[i] -= lr * m.Block.FF.B1Grad[i]
	}
	for i := range m.Block.FF.B2 {
		m.Block.FF.B2[i] -= lr * m.Block.FF.B2Grad[i]
	}
}

func clipGrads(model *Model, maxNorm float32) {
	for i := range model.ProjWGrad.Data {
		if model.ProjWGrad.Data[i] > maxNorm {
			model.ProjWGrad.Data[i] = maxNorm
		}
		if model.ProjWGrad.Data[i] < -maxNorm {
			model.ProjWGrad.Data[i] = -maxNorm
		}
	}
	for i := range model.Block.FF.W1Grad.Data {
		if model.Block.FF.W1Grad.Data[i] > maxNorm {
			model.Block.FF.W1Grad.Data[i] = maxNorm
		}
		if model.Block.FF.W1Grad.Data[i] < -maxNorm {
			model.Block.FF.W1Grad.Data[i] = -maxNorm
		}
	}
}
