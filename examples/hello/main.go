package main

import (
	"flag"
	"fmt"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/janpfeifer/must"
)

const (
	CoefficientMu    = 0.0 // μ_c
	CoefficientSigma = 5.0 //  σ_c
	BiasMu           = 1.0
	BiasSigma        = 10.0
)

var (
	flagNumExamples  = flag.Int("num_examples", 10000, "Number of examples to generate")
	flagNumFeatures  = flag.Int("num_features", 3, "Number of features")
	flagNoise        = flag.Float64("noise", 0.2, "Noise in synthetic data generation")
	flagNumSteps     = flag.Int("steps", 1000, "Number of gradient descent steps to perform")
	flagLearningRate = flag.Float64("lr", 0.1, "Initial learning rate.")
)

func main() {
	flag.Parse()

	//TrainMain()
	myTrainMain()
}

// AttachToLoop decorators. It will be redefined later.
// func AttachToLoop(loop *train.Loop) {
// 	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
// }

func AttachToLoop(loop *train.Loop) {
	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
	_ = plotly.New().Dynamic().ScheduleExponential(loop, 50, 1.1)
}

// TrainMain() does everything to train the linear model.
func TrainMain() {

	var backend = backends.New()

	// Select coefficients that we will try to predic.
	trueCoefficients, trueBias := initCoefficients(backend, *flagNumFeatures)
	fmt.Printf("Target: coefficients=%0.3v, bias=%0.3v\n", trueCoefficients.Value(), trueBias.Value())

	// Generate training data with noise.
	inputs, labels := buildExamples(backend, trueCoefficients, trueBias, *flagNumExamples, *flagNoise)
	fmt.Printf("Training data (inputs, labels): (%s, %s)\n\n", inputs.Shape(), labels.Shape())
	dataset := &TrivialDataset{"linear", []*tensors.Tensor{inputs}, []*tensors.Tensor{labels}}

	// Creates Context with learned weights and bias.
	ctx := context.New()
	ctx.SetParam(optimizers.ParamLearningRate, *flagLearningRate) // = "learning_rate"
	ctx.SetParam(regularizers.ParamL2, 1e-3)                      // 1e-3 of L2 regularization.

	// train.Trainer executes a training step.
	trainer := train.NewTrainer(backend, ctx, modelGraph,
		losses.MeanSquaredError,
		optimizers.StochasticGradientDescent(),
		nil, nil) // trainMetrics, evalMetrics
	loop := train.NewLoop(trainer)
	AttachToLoop(loop)

	// Loop for given number of steps. must.M1() panics, if loop.RunSteps returns an error.
	metrics := must.M1(loop.RunSteps(dataset, *flagNumSteps))
	_ = metrics // We are not interested in them in this example.

	// Print learned coefficients and bias -- from the weights in the dense layer.
	fmt.Println()
	coefVar, biasVar := ctx.GetVariableByScopeAndName("/dense", "weights"),
		ctx.GetVariableByScopeAndName("/dense", "biases")

	learnedCoef, learnedBias := coefVar.Value(), biasVar.Value()
	fmt.Printf("Learned: coefficients=%0.3v, bias=%0.3v\n", learnedCoef.Value(), learnedBias.Value())
}

func modelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	_ = spec // Not needed here, we know the dataset.
	logits := layers.DenseWithBias(ctx, inputs[0] /* outputDim= */, 1)
	return []*Node{logits}
}

// TrivialDataset always returns the whole data.
type TrivialDataset struct {
	name           string
	inputs, labels []*tensors.Tensor
}

var (
	// Assert Dataset implements train.Dataset.
	_ train.Dataset = &TrivialDataset{}
)

// Name implements train.Dataset.
func (ds *TrivialDataset) Name() string { return ds.name }

// Yield implements train.Dataset.
func (ds *TrivialDataset) Yield() (spec any, inputs, labels []*tensors.Tensor, err error) {
	return ds, ds.inputs, ds.labels, nil
}

// IsOwnershipTransferred tells the training loop that the dataset keeps ownership of the yielded tensors.
func (ds *TrivialDataset) IsOwnershipTransferred() bool {
	return false
}

// Reset implements train.Dataset.
func (ds *TrivialDataset) Reset() {}

func buildExamples(backend backends.Backend, coef, bias *tensors.Tensor, numExamples int, noise float64) (inputs, labels *tensors.Tensor) {
	e := NewExec(backend, func(coef, bias *Node) (inputs, labels *Node) {
		g := coef.Graph()
		numFeatures := coef.Shape().Dimensions[0]

		// Random inputs (observations).
		rngState := Const(g, RngState())
		rngState, inputs = RandomNormal(rngState, shapes.Make(coef.DType(), numExamples, numFeatures))
		coef = ExpandDims(coef, 0)

		// Calculate perfect labels.
		labels = ReduceAndKeep(Mul(inputs, coef), ReduceSum, -1)
		labels = Add(labels, bias)
		if noise > 0 {
			// Add some noise to the labels.
			var noiseVector *Node
			rngState, noiseVector = RandomNormal(rngState, labels.Shape())
			noiseVector = MulScalar(noiseVector, noise)
			labels = Add(labels, noiseVector)
		}
		return
	})
	examples := e.Call(coef, bias) //labels = inputs · coef + bias
	inputs, labels = examples[0], examples[1]
	return
}

// initCoefficients chooses random coefficients and bias. These are the true values the model will
// attempt to learn.
func initCoefficients(backend backends.Backend, numVariables int) (coefficients, bias *tensors.Tensor) {
	e := NewExec(backend, func(g *Graph) (coefficients, bias *Node) {
		rngState := Const(g, RngState())
		rngState, coefficients = RandomNormal(rngState, shapes.Make(dtypes.Float64, numVariables))
		coefficients = AddScalar(MulScalar(coefficients, CoefficientSigma), CoefficientMu) // θ=σ_c​⋅Z + μ_c
		rngState, bias = RandomNormal(rngState, shapes.Make(dtypes.Float64))
		bias = AddScalar(MulScalar(bias, BiasSigma), BiasMu) // b=σ_b​⋅Z + μ_b
		return
	})
	results := e.Call()
	coefficients, bias = results[0], results[1]
	return
}
