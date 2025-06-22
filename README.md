# Simple experiment for a little utility to draw diagrams of SDFs

Goal is to draw SDFs easily in python and provide some useful functions such that not everything has to be done in a shader

![image](output_image.png)

---

Requirements:
- Rust
- Python
- wgpu
- Maturin (py03) for python bindings

(not tested with different python/rust/wgpu versions or different machines yet. This was python 3.12 on a Macbook)

Instructions to Build:

From `/renderer`:

```
maturin develop
```

To run:

```
python script.py
```

Sample shape:

```python
canvas = Canvas(1024)

c = Circle(Point(0.125, 0.125), 0.125)
b = Box(Point(0, 0), 0.25, 0.25)
result = c | b
canvas.generate_image(result, "output_image.png")
```

This produces:

![image](simple.png)

---
This is what python produces and sends to webgpu for the first example:

`min(min(min(min(roundDifference(roundDifference(circle(p,vec2<f32>(-0.1000,-0.0750),0.3400),min(box(p-vec2<f32>(-0.1000,-0.0750),vec2<f32>(0.2000,0.3000)),box(p-vec2<f32>(-0.1000,-0.0750),vec2<f32>(0.7500,0.1000))), 0.05),circle(p,vec2<f32>(0.1720,0.1290),0.0200), 0.0075),circle(p,vec2<f32>(0.1720,0.1290),0.0100)),circle(p,vec2<f32>(0.4000,0.3000),0.0100)),line(p,vec2<f32>(0.4000,0.3000),vec2<f32>(0.1720,0.1290))),min(box(p-vec2<f32>(-0.1000,-0.0750),vec2<f32>(0.2000,0.3000)),box(p-vec2<f32>(-0.1000,-0.0750),vec2<f32>(0.7500,0.1000)))+0.01)`

It does *not* do any sort of [clever optimization](https://www.mattkeeter.com/research/mpr/). But, since it's purely to create simple images (and it's in a gpu shader anyway), performance doesn't really matter.