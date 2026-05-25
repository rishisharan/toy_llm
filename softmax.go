package main
import "math"

func softmax(x []float32) []float32 {

	out := make([]float32, len(x))
	max := x[0]
	for _, v := range x {
		if(v > max){
			max = v
		}
	}

	//e^(v-max) for each value
	var sum float32
	for i, v := range x {
		out[i] = float32(math.Exp(float64(v - max)))
		sum += out[i]
	}

	for i := range out {
		out[i] = out[i] / sum
	}

	return out
}