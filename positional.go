// positional.go
package main

import "math"

func PosEncoding(seqLen, embedDim int) Matrix {
    pe := NewMatrix(seqLen, embedDim)
    for pos := 0; pos < seqLen; pos++ {
        for i := 0; i < embedDim; i++ {
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

// adds positional encoding onto the embeddings in place
func AddPosEncoding(embeddings Matrix) Matrix {
    pe := PosEncoding(embeddings.Rows, embeddings.Cols)
    out := NewMatrix(embeddings.Rows, embeddings.Cols)
    for i := range out.Data {
        out.Data[i] = embeddings.Data[i] + pe.Data[i]
    }
    return out
}