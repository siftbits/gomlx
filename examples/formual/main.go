package main

import (
	"flag"
	"fmt"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/simplego"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

var (
	flagA            = flag.Float64("a", 1.0, "Value of a in the equation ax^2+bx+c")
	flagB            = flag.Float64("b", 2.0, "Value of b in the equation ax^2+bx+c")
	flagC            = flag.Float64("c", 4.0, "Value of c in the equation ax^2+bx+c")
	flagNumSteps     = flag.Int("steps", 10, "Number of gradient descent steps to perform")
	flagLearningRate = flag.Float64("lr", 0.1, "Initial learning rate.")
)

// f(x) = ax^2 + bx + c
func f(x *Node) *Node {
	f := MulScalar(Square(x), *flagA)
	f = Add(f, MulScalar(x, *flagB))
	f = AddScalar(f, *flagC)
	return f
}

var round int

// minimizeF does one gradient descent step on F by moving a variable "x",
// and returns the value of the function at the current "x".
func minimizeF(ctx *context.Context, graph *Graph) *Node {
	// Create or reuse existing variable "x" -- no graph operation is created with this, it's
	// only a reference.
	xVar := ctx.VariableWithShape("x", shapes.Make(dtypes.Float64))

	x := xVar.ValueGraph(graph) // Read variable for the current graph.
	y := f(x)                   // Value of f(x).

	// Gradient always return a slice, we take the first element for grad of X.
	gradX := Gradient(y, x)[0]

	// 以下为optism内容
	// stepNum += 1
	stepNumVar := ctx.VariableWithValue("stepNum", 0.0) // Creates the variable if not existing, or retrieve it if already exists.
	stepNum := stepNumVar.ValueGraph(graph)
	stepNum = OnePlus(stepNum)
	stepNumVar.SetValueGraph(stepNum)

	// step = -learningRate * gradX / Sqrt(stepNum)
	step := Div(gradX, Sqrt(stepNum))
	step = MulScalar(step, -*flagLearningRate)

	// x += step
	x = Add(x, step)
	xVar.SetValueGraph(x)

	//	xVal := ctx.InspectVariable(ctx.Scope(), "x").Value()
	stepVal := ctx.InspectVariable(ctx.Scope(), "stepNum").Value()

	// yVar := ctx.VariableWithValue("y", 0.0) // Creates the variable if not existing, or retrieve it if already exists.
	// yVar.SetValueGraph(y)

	// yVal := ctx.InspectVariable(ctx.Scope(), "y").Value()

	// fmt.Printf("Round: %d at x=%g, f(x)=%g   steps=%g.\n", round, xVal.Value(), yVal.Value(), stepVal.Value())

	//fmt.Printf("Round: %d at x=%g,    steps=%g.\n", round, xVal.Value(), stepVal.Value())

	//	fmt.Printf("Round: %d at x=%g,    \n", round, xVal.Value())
	fmt.Printf("Round: %d  steps=%g.\n", round, stepVal.Value())
	return y // f(x)
}

func Solve(b backends.Backend) {
	ctx := context.New()
	exec := context.NewExec(b, ctx, minimizeF)
	for ii := 0; ii < *flagNumSteps-1; ii++ {
		y := exec.Call()[0]

		x := ctx.InspectVariable(ctx.Scope(), "x").Value()
		stepNum := ctx.InspectVariable(ctx.Scope(), "stepNum").Value()

		fmt.Printf(" round at x=%g, f(x)=%g after %d steps.\n", x.Value(), y.Value(), int32(stepNum.Value().(float64)))
	}
	y := exec.Call()[0]
	x := ctx.InspectVariable(ctx.Scope(), "x").Value()
	stepNum := ctx.InspectVariable(ctx.Scope(), "stepNum").Value()
	fmt.Printf("Minimum found at x=%g, f(x)=%g after %f steps.\n", x.Value(), y.Value(), stepNum.Value())
}

func main() {
	flag.Parse()

	b := backends.MustNew()
	Solve(b)

}
