package main


func tokenize(word string) (map[string]int, map[int]string) {
	encodeMap :=encode(word)
	decodeMap :=decode(encodeMap)
	return encodeMap, decodeMap
}

func encode(word string) map[string]int {
	encode := map[string]int{};
	index := 0
	for _, ch := range word {
		_, exists := encode[string(ch)]
		if !exists {
			encode[string(ch)] = index
			index++
		}
	}
	return encode
}
func decode(encode map[string]int) map[int]string {
    decode := map[int]string{}
    for ch, id := range encode {
        decode[id] = ch
    }
    return decode
}