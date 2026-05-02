// model.go
package main

type Model struct {
	Embedding    Embedding
	Block        *TransformerBlock
	ProjW        Matrix
	ProjWGrad    Matrix
	lastX        Matrix
	lastLogits   Matrix
	lastTokenIDs []int
}

func NewModel(vocabSize, embedDim, hiddenDim int) Model {
	proj := NewMatrix(embedDim, vocabSize)
	for i := range proj.Data {
		proj.Data[i] = (randFloat() - 0.5) * 0.01
	}
	block := NewTransformerBlock(embedDim, hiddenDim)
	return Model{
		Embedding: NewEmbedding(vocabSize, embedDim),
		Block:     &block,
		ProjW:     proj,
		ProjWGrad: NewMatrix(embedDim, vocabSize),
	}
}

func (m *Model) Forward(tokenIDs []int) Matrix {  // ← * here
	m.lastTokenIDs = tokenIDs
	x := m.Embedding.Forward(tokenIDs)
	x = AddPosEncoding(x)
	x = m.Block.Forward(x)
	m.lastX = x
	logits := MatMul(x, m.ProjW)
	m.lastLogits = logits
	return logits
}

func (m *Model) Backward(targets []int) {
	dLogits := CrossEntropyBackward(m.lastLogits, targets)
	m.ProjWGrad = MatMul(Transpose(m.lastX), dLogits)
	dX := MatMul(dLogits, Transpose(m.ProjW))
	m.Block.Backward(dX)
	m.Embedding.Backward(m.Block.lastInputGrad)
}

