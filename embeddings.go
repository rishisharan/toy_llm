package main

type Embedding struct {
	Weight Matrix
}

func NewEmbedding(vocabSize, embedDim int) Embedding {
	w := newMatrix(vocabSize, embedDim)
	for i := range w.Data {
		w.Data[i] = (randFloat() - 0.5) * 0.01
	}
	return Embedding{Weight: w}
}

func (e Embedding) Forward(tokenIDs []int) Matrix {
	out := newMatrix(len(tokenIDs), e.Weight.Cols)
	for i, id := range tokenIDs {
		for j := 0; j < e.Weight.Cols; j++ {
			out.Set(i, j, e.Weight.Get(id, j))
		}
	}
	return out
}
