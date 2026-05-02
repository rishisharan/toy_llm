// embeddings.go
package main

type Embedding struct {
    Weight Matrix // shape: (vocabSize x embedDim)
    WeightGrad Matrix
    lastIDs    []int
}

func NewEmbedding(vocabSize, embedDim int) Embedding {
    w := NewMatrix(vocabSize, embedDim)
    for i := range w.Data { w.Data[i] = (randFloat() - 0.5) * 0.01 }
    return Embedding{Weight: w, WeightGrad: NewMatrix(vocabSize, embedDim)}
}

func (e *Embedding) Forward(tokenIDs []int) Matrix {
    e.lastIDs = tokenIDs
    out := NewMatrix(len(tokenIDs), e.Weight.Cols)
    for i, id := range tokenIDs {
        for j := 0; j < e.Weight.Cols; j++ {
            out.Set(i, j, e.Weight.Get(id, j))
        }
    }
    return out
}

func (e *Embedding) Backward(dOut Matrix) {
    // scatter gradients back to the embedding rows
    for i, id := range e.lastIDs {
        for j := 0; j < dOut.Cols; j++ {
            e.WeightGrad.Data[id*dOut.Cols+j] += dOut.Get(i, j)
        }
    }
}