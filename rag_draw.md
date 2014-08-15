
Region Adjacency Graphs Drawing
-------------------------------

A lot of Image Processing algorithms are based on intuition from visual cues.
Region Adjacency Graphs would also benefit if they were somehow drawn back on
the images they represent. If we are able to see the nodes, edges, and the edges
weights, we can fine tune our parameters and algorithms to suit our needs. I had
written a small hack in this [blog
post](http://vcansimplify.wordpress.com/2014/07/06/scikit-image-rag-
introduction/) to help better visualize the results. Later,
[Juan](http://www.janelia.org/people/juan-nunez-iglesias) suggested I port if
for scikit-image. It will indeed be a very helpful tool for anyone who wants to
explore RAGs in scikit-image.

## Getting Started

You will need to pull for [this](https://github.com/scikit-image/scikit-
image/pull/1087) Pull Request to be able to execute the code below. I'll start
by defining a custom `show_image` function to aid displaying in IPython
notebooks.


```python
from skimage import graph, data, io, segmentation, color
from matplotlib import pyplot as plt
from skimage.measure import regionprops
import numpy as np
from matplotlib import colors

def show_image(img):
    width = img.shape[1]/50.0
    height = img.shape[0]*width/img.shape[1]
    f = plt.figure(figsize=(width, height))
    plt.imshow(img)
```

We will start by loading a demo image just containing 3 bold colors to help us
see how the `draw_rag` function works.

```python
image = io.imread('/home/vighnesh/Desktop/images/colors.png')
show_image(image)
```

![png](rag_draw_files/rag_draw_3_0.png)


We will now use the SLIC algorithm to give us an over-segmentation, on which we
will build our RAG.

```python
labels = segmentation.slic(image, compactness=30, n_segments=400)
```
Here's what the over-segmentation looks like.

```python
border_image = segmentation.mark_boundaries(image, labels, (0,0,0))
show_image(border_image)
```

![png](rag_draw_files/rag_draw_7_0.png)

## Drawing the RAGs
We can now form out RAG and see how it looks.

```python
rag = graph.rag_mean_color(image, labels)
out = graph.draw_rag(labels, rag, border_image)
show_image(out)
```

![png](rag_draw_files/rag_draw_9_1.png)


In the above image, nodes are shown in yellow whereas edges are shown in green.
Each region is represented by its centroid. As Juan pointed out, many edges will be 
difficult to see because of low contrant between them and the image, as seen above. To
counter this we support the `desaturate` option. When set to `True` the image is 
converted to grayscale before displaying. Hence all the image pixels are a shade of gray,
while the edges and nodes stand out.

```python
out = graph.draw_rag(labels, rag, border_image, desaturate=True)
show_image(out)
```

![png](rag_draw_files/rag_draw_11_0.png)


Although the above image does very well to show us individual regions and their
adjacency relationships, it does nothing to show us the magnitude of edges. To
give us more information about the magnitude of edges, we have the `colormap`
option. It colors edges between the first and the second color depending on
their weight.

```python
blue_red = colors.ListedColormap(['blue','red'])
out = graph.draw_rag(labels, rag, border_image, desaturate=True,colormap=blue_red)
show_image(out)
```

![png](rag_draw_files/rag_draw_13_0.png)


As you can see, the edges between similar regions are blue, whereas edges
between dissimilar regions are red. `draw_rag` also accepts a `thresh` option.
All edges above `thresh` are not considered for drawing.


```python
out = graph.draw_rag(labels, rag, border_image, desaturate=True,colormap=blue_red, thresh=10)
show_image(out)
```

![png](rag_draw_files/rag_draw_15_0.png)


Another clever trick is to supply a blank image, this way, we can see the RAG
unobstructed.

```python
cyan_red = colors.ListedColormap(['cyan','red'])
out = graph.draw_rag(labels, rag, np.zeros_like(image), desaturate=True,colormap=cyan_red)
show_image(out)
````

![png](rag_draw_files/rag_draw_17_0.png)
**Ahhh, magnificent.** 

Here is a small piece of code which produces a typical deaturated color-distance RAG.

```python
image = data.coffee()
labels = segmentation.slic(image, compactness=30, n_segments=400)
rag = graph.rag_mean_color(image, labels)
cmap = colors.ListedColormap(['blue','red'])
out = graph.draw_rag(labels, rag, image,border_color=(0,0,0), desaturate=True,colormap=cmap)
show_image(out)
```

![png](rag_draw_files/coffee_extra.png)
If you notice the above image, you will find some dges crossing over each other. This is because, some regions are convex. Hence their centroid lies outside their boundary and edges eminating from it can cross other edges.


## Examples
I will go over some examples of RAG drawings, since most of it is similar, I won't repeat the 
code here. The Ncut technique, wherever used, was with its default parameters.

### Color distance RAG of Coffee on black background
![png](rag_draw_files/cup1.png)

### Color distance RAG of Coffee after applying NCut
![png](rag_draw_files/cup2.png)
Notice how the the centroid of the white rim of the cup is placed at its centre. It is the one adjacent to the centroid of the gray region of the upper part of the spoon, connected to it via a blue edge. Notice how this edge crosses others.

### Color distance RAG of Lena
![png](rag_draw_files/lena.png)

### A futuristic car and its color distance RAG after NCut
![jpg](rag_draw_files/car.jpg)


![png](rag_draw_files/car.png)

### Coins Image and their color distance RAG after NCut
![png](rag_draw_files/coins.png)


## Further Improvements
 - A point that was brought up in the PR as well is that thick lines would immensely enhance the visual
appeal of the output. As and when they are implemented, `rag_draw` should be modified to support drawing
thick edges.
 - As centroids don't always lie in within an objects boundary, we can represent regions by a point other than their centroid, something which always lies within the boundary. This would allow for better visualization of the actual RAG from its drawing. 
