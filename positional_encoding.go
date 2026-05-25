package main
import "math"

func positionalEncoding(seqLen, embedDim int) Matrix {
	pe := newMatrix(seqLen, embedDim)
	for pos := 0; pos<seqLen; pos++ {
		for i :=0; i<embedDim;i++ {
			angle := float64(pos) / math.Pow(10000, float64(i)/float64(embedDim))
			if i%2 == 0 {
				pe.Set(pos, i, float32(math.Sin(angle)))
			} else {
				pe.Set(pos, i, float32(math.Cos(angle)))
			}
		}
	}
	return pe
}

func addPositionalEncoding(embeddings Matrix) Matrix {
    pe := positionalEncoding(embeddings.Rows, embeddings.Cols)
    out := newMatrix(embeddings.Rows, embeddings.Cols)
    for i := range out.Data {
        out.Data[i] = embeddings.Data[i] + pe.Data[i]
    }
    return out
}