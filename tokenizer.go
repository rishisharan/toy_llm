package main

type Tokenizer struct {
    StrToID map[string]int
    IDToStr map[int]string
    Size    int
}

func NewTokenizer(text string) Tokenizer {
    vocab := make(map[string]int)
    reverse := make(map[int]string)
    id := 0
    for _, ch := range text {
        ch := string(ch)
        if _, exists := vocab[ch]; !exists {
            vocab[ch] = id
            reverse[id] = ch
            id++
        }
    }
    return Tokenizer{StrToID: vocab, IDToStr: reverse, Size: id}
}

func (t Tokenizer) Encode(text string) []int {
    ids := make([]int, 0, len(text))
    for _, ch := range text {
        ids = append(ids, t.StrToID[string(ch)])
    }
    return ids
}

func (t Tokenizer) Decode(ids []int) string {
    out := ""
    for _, id := range ids {
        out += t.IDToStr[id]
    }
    return out
}