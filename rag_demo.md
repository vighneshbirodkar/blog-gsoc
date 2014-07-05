
#scikit-image RAG Introduction

Humans possess an incredible ability to identify objects in an image. Image
processing algorithms are still far behind this ability. **Segmentation** is the
process of identifying objects in an image. All pixels belonging to an object
should get a unique label in an ideal segmentation.

In the example below, we will see how Region Adjacency Graphs (RAGs) attempt to
solve this problem.


# Getting Started
The function `show_img` is defined so that images displayed are big enough for
comfortable viewing. We start with `img`, a nice fresh image of a coffee cup.

```python
from skimage import graph, data, io, segmentation, color
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from skimage import draw
import numpy as np


def show_img(img):
    width = 10.0
    height = img.shape[0]*width/img.shape[1]
    f = plt.figure(figsize=(width, height))
    plt.imshow(img)

img = data.coffee()
show_img(img)

```
![png](rag_demo_files/rag_demo_2_0.png)


# Over Segmentation
We segment the image using SLIC algorithm. The segmentation algorithm will
assign one unique label to what it perceives as a **Region**. This is a
localized cluster of pixels sharing some similar proerpty, in this case their
color. The label of each pixel is stored in the `labels` array.

`regionprops` helps us compute various features of these regions, the one we
will be using (purely for visualization) is the centroid.


    labels = segmentation.slic(img, compactness=30, n_segments=400)
    labels = labels + 1  # So that no labelled region is 0 and ignored by regionprops
    regions = regionprops(labels)


The `label2rgb` function assigns a specific color to all pixels belonging to one
region (having the same label). In this case, in `label_rgb` each pixel is
replaces with the average `RGB` color of its region.


    label_rgb = color.label2rgb(labels, img, kind='avg')
    show_img(label_rgb)



![png](rag_demo_files/rag_demo_6_0.png)


Just for clarity, we use `mark_boundaries` to highlight the region boundaries.
You will notice the the image is divided into more regions than required. This
phenomenon is called **Over-Segmentation**.


    label_rgb = segmentation.mark_boundaries(label_rgb, labels, (0, 0, 0))
    show_img(label_rgb)


![png](rag_demo_files/rag_demo_8_0.png)


# Enter, RAGs

Region Adjacency Graphs, as the name suggests represent adjacency of regions
with a graph. Each region in the image is a node in a graph. There is an edge
between every pair of adjacent regions (regions whose pixels are adjacent). The
weight of between every two nodes can be defined in a variety of ways. For this
example, we will use the difference of average color between two regions as
their edge weight. The more similar the regions, the lesser the weight between
them. Because we are using difference in mean color to compute the edge weight,
the method has been named `rag_mean_color`.


    rag = graph.rag_mean_color(img, labels)

For our visualization, we are also adding an additional property to a node, the
coordinated of its centroid.


    for region in regions:
        rag.node[region['label']]['centroid'] = region['centroid']

`display_edges` is a function to draw the edges of a RAG on it's cooresponsing
image. The edges are drawn in green. It also marks the centroid of each region
by a yellow dot. It also takes an argument `thresh`, only edges with weight
lower than `thresh` are drawn.


    def display_edges(img, rag, thresh):
        img = img.copy()
        for edge in rag.edges_iter(data=True):
    
            r1,r2,data = edge
            y1 = rag.node[r1]['centroid'][0]
            x1 = rag.node[r1]['centroid'][1]
    
            y2 = rag.node[r2]['centroid'][0]
            x2 = rag.node[r2]['centroid'][1]
            x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2)
            line  = draw.line(y1,x1,y2,x2)
            circle = draw.circle(y1,x1,2)
    
            wt = data['weight']
            
            if wt < thresh :
                img[line] = 0,1,0
            img[circle] = 1,1,0
    
        return img

We call the function with `thresh = infinity` so that all edges are drawn. I
myself was surprised with the beauty of the following output.


    edges_drawn_all = display_edges(label_rgb, rag, np.inf )
    show_img(edges_drawn_all)


![png](rag_demo_files/rag_demo_16_0.png)


Let's see what happens by setting `thresh` to `30`, a value I arrived at with
some trial and error.


    edges_drawn_30 = display_edges(label_rgb, rag, 30 )
    show_img(edges_drawn_30)



![png](rag_demo_files/rag_demo_18_0.png)


### Alas, the graph is cut

As you can see above, the RAG is now divided into disconnected regions. If you
notice, the table above and to the right of the dish is one big connected
component.

# Threshold Cut

The function `cut_threshold` removes edges below a soecified threshold and then
labels a connected component as one region. Once the RAG is constructed, many
such strategies can be employed to improved the segmentation. Thresholding
however, requires human assistance.


    final_labels = graph.cut_threshold(labels, rag, 30)
    final_label_rgb = color.label2rgb(final_labels, img, kind='avg')
    final_label_rgb = segmentation.mark_boundaries(final_label_rgb, final_labels, (0,0,0))
    show_img(final_label_rgb)


![png](rag_demo_files/rag_demo_22_0.png)


### Not perfect, but not that bad I would say


    
