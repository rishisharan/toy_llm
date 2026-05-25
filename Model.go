package main

type Model struct {
	Embedding *Embedding
	FF        *FeedForward
	ProjW     Matrix
}

func NewModel(vocabSize, embedDim, hiddenDim int) Model {
	// create ProjW matrix with random values
	ProjW := newMatrix(embedDim, vocabSize)
	for i := range ProjW.Data {
		ProjW.Data[i] = (randFloat() - 0.5) * 0.1
	}
	ff := NewFeedFroward(embedDim, hiddenDim)
	emb := NewEmbedding(vocabSize, embedDim)
	return Model{
		Embedding: &emb,
		FF:        &ff,
		ProjW:     ProjW,
	}
}
