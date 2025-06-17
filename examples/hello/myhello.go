package main

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	g "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/janpfeifer/must"
)

func myTrainMain() {

	var backend = backends.New()
	trueCoef, trueBias := initCoefs(backend, *flagNumFeatures)
	fmt.Printf("Target: coefficients=%0.3v, bias=%0.3v\n", trueCoef.Value(), trueBias.Value())

	// Generate training data with noise.
	inputs, labels := genExamples(backend, trueCoef, trueBias, *flagNumExamples, *flagNoise)
	fmt.Println("aaaaa", inputs.Shape(), inputs.Size(), inputs.Value().([][]float64)[0])
	fmt.Printf("Training data (inputs, labels): (%s, %s)\n\n", inputs.Shape(), labels.Shape())

	ds := &trivialDataset{
		name:   "linear",
		inputs: []*tensors.Tensor{inputs},
		labels: []*tensors.Tensor{labels},
	}

	ctx := context.New()
	ctx.SetParam(optimizers.ParamLearningRate, *flagLearningRate) // = "learning_rate"
	ctx.SetParam(regularizers.ParamL2, 1e-3)

	trainer := train.NewTrainer(
		backend,
		ctx,
		modelGraph,
		losses.MeanSquaredError,
		optimizers.StochasticGradientDescent(),
		nil,
		nil)

	loop := train.NewLoop(trainer)
	commandline.AttachProgressBar(loop)

	metrics := must.M1(loop.RunSteps(ds, *flagNumSteps))

	for _, v := range metrics {
		fmt.Println("metrics", v.String(), v.Value())
	}

	fmt.Println()
	coefVar := ctx.GetVariableByScopeAndName("/dense", "weights")
	biasVar := ctx.GetVariableByScopeAndName("/dense", "biases")

	learnedCoef, learnedBias := coefVar.Value(), biasVar.Value()
	fmt.Printf("Learned: coefficients=%0.3v, bias=%0.3v\n", learnedCoef.Value(), learnedBias.Value())
}

func initCoefs(b backends.Backend, varNums int) (coef, bias *tensors.Tensor) {
	graphFn := func(gh *g.Graph) (coef, bias *g.Node) {
		rngState := g.Const(gh, g.RngState())

		rngState, coef = g.RandomNormal(rngState, shapes.Make(dtypes.Float64, varNums))
		coef = g.AddScalar(g.MulScalar(coef, CoefficientSigma), CoefficientMu) // θ=σ_c​⋅Z + μ_c

		_, bias = g.RandomNormal(rngState, shapes.Make(dtypes.Float64))
		bias = g.AddScalar(g.MulScalar(bias, BiasSigma), BiasMu) // b=σ_b​⋅Z + μ_b
		return
	}

	results := g.ExecOnceN(b, graphFn)
	coef = results[0]
	bias = results[1]
	return
}

func genExamples(b backends.Backend, coef, bias *tensors.Tensor, sampleNum int, noise float64) (inputs, labels *tensors.Tensor) {
	graphFn := func(coef, bias *g.Node) (inputs, labels *g.Node) {
		gh := bias.Graph()
		featureNum := coef.Shape().Dimensions[0]

		fmt.Println("111111", coef.Shape())

		rngState := g.Const(gh, g.RngState())

		rngState, inputs = g.RandomNormal(rngState, shapes.Make(coef.DType(), sampleNum, featureNum))
		coef = g.InsertAxes(coef, 0)

		fmt.Println("22222", coef.Shape())

		labels = g.ReduceAndKeep(g.Mul(inputs, coef), g.ReduceSum, -1) // An axis set to -1 is converted to `rank - 1`.
		labels = g.Add(labels, bias)
		if noise > 0 {
			var noiseVec *g.Node
			_, noiseVec = g.RandomNormal(rngState, labels.Shape())
			noiseVec = g.MulScalar(noiseVec, noise)
			labels = g.Add(labels, noiseVec)
		}

		fmt.Println("3333333", coef.Shape(), inputs.Shape(), labels.Shape())
		return
	}
	results := g.ExecOnceN(b, graphFn, coef, bias)
	inputs, labels = results[0], results[1]
	return
}

type trivialDataset struct {
	name   string
	inputs []*tensors.Tensor
	labels []*tensors.Tensor
}

var _ train.Dataset = &trivialDataset{}

func (td *trivialDataset) Name() string { return td.name }

func (td *trivialDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	return td, td.inputs, td.labels, nil
}

// Reset restarts the dataset from the beginning. Can be called after io.EOF is reached,
// for instance when running another evaluation on a test dataset.
func (td *trivialDataset) Reset() {

}

// IsOwnershipTransferred tells the training loop that the dataset keeps ownership of the yielded tensors.
func (td *trivialDataset) IsOwnershipTransferred() bool {
	return false
}
