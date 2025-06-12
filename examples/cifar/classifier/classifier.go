// Package classifier is a Cifar-10 classifier.
// It loads a pre-trained model and offers a Classify method that will classify any image,
// by first resizing it to the model's input size.
//
// To use it, create a Classifier with New(), and then simply call its Classify method with a 32x32 image.
//
// This is an example of how to serve a model for inference.
package main

import (
	"flag"
	"fmt"
	"image"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/examples/cifar"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/tensors/images"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagImgDir     = flag.String("img", "/opt/lewgun/source/go/github.com/zjtv/gomlx/examples/cifar/img", "Directory to cache the images.")
	flagCheckpoint = flag.String("checkpoint", "~/work/cifar/checkpoint", "Directory save and load checkpoints from. If left empty, no checkpoints are created.")
)

var C10Labels = []string{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}

// Classifier holds the Cifar-10 model compiled.
// It will use XLA with GPU if available or CPU by default. But the backend can be configured with GOMLX_BACKEND.
type Classifier struct {
	// backend is created with defaults, which uses GOMLX_BACKEND if it is set.
	backend backends.Backend

	// ctx with the model's weights.
	ctx *context.Context

	// exec is used to execute the model with a context.
	exec *context.Exec
}

// New creates a new Classifier object that can be used to classify images using a Cifar-10 model.
func New(checkpointDir string) (*Classifier, error) {
	c := &Classifier{
		backend: backends.MustNew(),
		ctx:     context.New(),
	}

	// Notice all hyperparameters are read from the checkpoint as well, so it will be build the
	// same model.
	// We don't need to keep the checkpoint handler around, since we are not going to use it to save.
	_, err := checkpoints.Load(c.ctx).
		Dir(checkpointDir).
		Done()
	if err != nil {
		return nil, errors.WithMessagef(err, "failed while loading CIFAR-10 model from %q", checkpointDir)
	}
	c.ctx = c.ctx.Reuse() // Mark it to reuse variables: it will be an error to create a new variable -- for extra sanity checking.

	modelFn, err := cifar.SelectModelFn(c.ctx)
	if err != nil {
		return nil, errors.WithMessagef(err, "cannot build model from checkpoint %q, invalid model type", checkpointDir)
	}

	// Create model executor.
	c.exec = context.NewExec(c.backend, c.ctx.In("model"), func(ctx *context.Context, image *graph.Node) (choice *graph.Node) {
		// We take the first result from the modelFn -- it returns a slice.
		image = graph.ExpandAxes(image, 0) // Create a batch dimension of size 1.
		logits := modelFn(ctx, nil, []*graph.Node{image})[0]
		// Take the class with highest logit value.
		choice = graph.ArgMax(logits, -1, dtypes.Int32)
		// Remove batch dimension.
		choice = graph.Reshape(choice) // No dimensions given, means a scalar.
		return
	})
	return c, nil
}

// Classify takes a 32x32 image and returns a Cifar-10 classification according to the models.
// Use cifar.C10Labels to convert the returned class to a string name.
// The returned class is from 0 to 9.
//
// TODO: resize image to 32x32, used by the model.
func (c *Classifier) Classify(img image.Image) (int32, error) {
	input := images.ToTensor(dtypes.Float32).Single(img)
	var outputs []*tensors.Tensor
	err := exceptions.TryCatch[error](func() { outputs = c.exec.Call(input) })
	if err != nil {
		return 0, err
	}
	classID := tensors.ToScalar[int32](outputs[0]) // Convert tensor to Go value.
	return classID, nil
}

func main() {
	flag.Parse()

	c, _ := New(*flagCheckpoint)

	// Classify each .png image found in the given directory
	entries, err := os.ReadDir(*flagImgDir)
	if err != nil {
		panic(err)
	}
	for _, path := range entries {
		if !path.IsDir() && strings.HasSuffix(path.Name(), ".png") {
			imgpath := filepath.Join(*flagImgDir, path.Name())
			f, err := os.Open(imgpath)
			if err != nil {
				log.Fatal("open error:", err)
			}
			defer f.Close()

			img, err := png.Decode(f)
			if err != nil {
				log.Fatal("decode error:", err)
			}
			n, _ := c.Classify(img)

			fmt.Printf("path=%q, n=%d, label=%q\n", imgpath, n, C10Labels[n])
		}
	}

}
