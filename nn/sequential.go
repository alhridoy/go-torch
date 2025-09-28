package nn

import "go-torch/tensor"


//a container for layers arranged in a sequential order.
type Sequential struct {
	layers []Layer
}


func NewSequential() *Sequential {
	return &Sequential{
		layers: make([]Layer, 0),
	}
}


// adds a new layer to the sequential model.
func (s *Sequential) Add(layer Layer) {
	s.layers = append(s.layers, layer)
}


// performs the forward pass for the entire sequence of layers.
func (s *Sequential) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	var err error
	for _, layer := range s.layers {
		x, err = layer.Forward(x)
		if err != nil {
			return nil, err
		}
	}
	return x, nil
}


// returns a slice of all parameters from all layers in the model.
func (s *Sequential) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{}
	for _, layer := range s.layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}


// calls ZeroGrad on all layers in the model.
func (s *Sequential) ZeroGrad() {
	for _, layer := range s.layers {
		layer.ZeroGrad()
	}
}


func (s *Sequential) Layers() []Layer {
	return s.layers
}


// sets the model and all its layers to training mode.
func (s *Sequential) Train() {
	for _, layer := range s.layers {
		layer.Train()
	}
}

// sets the model and all its layers to evaluation mode.
func (s *Sequential) Eval() {
	for _, layer := range s.layers {
		layer.Eval()
	}
}