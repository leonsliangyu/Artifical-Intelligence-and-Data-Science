'''
@author: Devangini
'''

'''
    Liangyu Wang
    980025288
'''

import pydot 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class TreePlot:
    """
    This class creates tree plot for search tree
    """
    
    def __init__(self):
        """
        Constructor
        """
        # create graph object
        self.graph = pydot.Dot(graph_type='graph', dpi = 500)
        #index of node
        self.index = 0
        
    
    def createGraph(self, node, currentNode):
        """
        This method adds nodes and edges to graph object
        Similar to printTree() of Node class
        """
        
        # assign hex color
        if node.state == currentNode.state:
            color = "#ee0011"
        elif node.fringe:
            color = "#1895f3"
        else:
            color = "#00ee11"
            
        #create node
        parentGraphNode = pydot.Node(str(self.index) + " " + \
            node.state, style="filled", \
            fillcolor = color)
        self.index += 1
        
        #add node
        self.graph.add_node(parentGraphNode)
        
        #call this method for child nodes
        for childNode in node.children:
            childGraphNode = self.createGraph(childNode, currentNode)
            
            #create edge
            edge = pydot.Edge(parentGraphNode, childGraphNode)
            
            #add edge
            self.graph.add_edge(edge)
            
        return parentGraphNode
        
    
    def generateDiagram(self, rootNode, currentNode):
        """
        This method generates diagram
        """
        #add nodes to edges to graph
        self.createGraph(rootNode, currentNode)
        
        #show the diagram
        self.graph.write_png('graph.png')
        img=mpimg.imread('graph.png')
        plt.imshow(img)
        plt.axis('off')
 #       mng = plt.get_current_fig_manager()
#        mng.window.state('zoomed')
        plt.show()
  
    
