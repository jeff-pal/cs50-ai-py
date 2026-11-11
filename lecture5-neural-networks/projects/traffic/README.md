# Traffic Sign Recognition — Experimentation Process

To pick a final architecture for `get_model`, I wrote a small throwaway script that
loaded the full GTSRB dataset once (26,640 images across 43 categories), cached it,
and then trained/evaluated eight different `Sequential` architectures back-to-back
for 10 epochs each with an identical 60/40 train/test split, so the comparisons would
be apples-to-apples. The results (test accuracy after 10 epochs):

| # | Architecture | Params | Train time | Test accuracy |
|---|---|---|---|---|
| 1 | 1 conv(32)+pool → dense(128) | 809,387 | 25.5s | 96.48% |
| 2 | 1 conv(32), **no pooling** → dense(128) | 3,217,835 | 61.1s | 95.88% |
| 3 | 2× conv(32)+pool → dense(128) | 163,275 | 24.4s | 97.74% |
| 4 | conv(32)+pool → conv(64)+pool → dense(128) | 319,979 | 32.0s | 97.86% |
| 5 | same as #4 + dropout(0.5) | 319,979 | 29.9s | 98.12% |
| 6 | same as #5, dense(256) instead of 128 | 620,523 | 37.9s | **98.72%** |
| 7 | same as #5, dense(16) instead of 128 | 57,003 | 25.9s | 61.10% |
| 8 | same as #5 but dropout(0.7) | 319,979 | 29.9s | 98.09% |

**What worked well:** Stacking two convolution+pooling blocks (32 filters, then 64)
clearly beat a single block, both in accuracy and in parameter count — pooling after
each convolution lets the network build up to larger receptive fields while actually
*shrinking* the data instead of carrying a huge flattened feature map into the dense
layers. 

Adding dropout (0.5) before the output layer also reliably helped test
accuracy, and increasing the hidden dense layer from 128 to 256 units squeezed out a
bit more (model #6, which I ended up using in `get_model`: two conv/pool blocks with
32 then 64 filters, a 256-unit dense layer, dropout 0.5, and a 43-unit softmax
output). Running the full `traffic.py` pipeline with this architecture on the real
dataset reached **99.04% test accuracy** with a final loss of 0.043.

**What didn't work well:** Removing pooling entirely (model #2) was the biggest
surprise — it tripled the parameter count and more than doubled training time per
epoch, yet accuracy actually got slightly *worse*, presumably because the dense layer
was now trying to make sense of a much larger, noisier flattened representation
instead of the compact, spatially-pooled one. The other clear failure was making the
hidden dense layer too narrow (model #7, only 16 units): accuracy collapsed to 61%
and training accuracy itself never rose much above 22%, meaning the network simply
didn't have enough capacity in that layer to represent 43 classes — dropout on top of
an already-narrow bottleneck starved the model further. Cranking dropout up to 0.7
(model #8) didn't hurt final test accuracy much, but it visibly slowed down how
quickly training accuracy climbed across epochs, which makes sense since a bigger
fraction of the dense layer's units are zeroed out on every pass.

**What I noticed along the way:** The single biggest lever wasn't the architecture at
all — it was *preprocessing*. My first fully-wired end-to-end run of `traffic.py`
(before I normalized pixel values in `load_data`) started training with a loss around
123 on the very first batch, because raw 0–255 pixel values produce huge activations
and gradients. Simply converting each resized image to `float32` and dividing by
255.0 in `load_data` dropped the first-batch loss to about 3.7 and raised the same
architecture's final test accuracy from 96.16% to 99.04% with no other changes. I
also noticed that GTSRB images come in a wide range of native resolutions (I spot
checked several `.ppm` files, with shapes anywhere from roughly 25×25 up to 200×200+),
so resizing with `cv2.resize` to a fixed `IMG_WIDTH × IMG_HEIGHT` is doing real work
to make the inputs uniform, and that even a fairly small 30×30×3 input is enough for
a small CNN to tell 43 sign categories apart with very high accuracy once the pixel
values are properly scaled and the network has at least two pooling stages to keep
the hidden representation compact.
