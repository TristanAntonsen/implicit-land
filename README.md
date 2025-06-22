# Simple experiment for a little utility to draw diagrams of SDFs

Goal is to draw SDFs easily and provide some useful functions such that not everything has to be done in a shader

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