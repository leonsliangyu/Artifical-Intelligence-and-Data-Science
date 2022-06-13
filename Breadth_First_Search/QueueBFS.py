'''
@author: Devangini
'''

'''
    Liangyu Wang
    980025288
'''

from Node import Node
from collections import deque
from GraphData import graph
from TreePlot import TreePlot
import time

"""
   This function performs BFS search using a queue
"""

def BFS_Liangyu(graph=None, first_student=None, second_student=None):
    
    #check for missing arguments
    if graph is None:
        print("invalid graph")
        return
    if first_student is None:
        print("invalid first student")
        return
    if second_student is None:
        print("invalid second student")
        return
    
    
    # check if given student is in the graph, if not return a message
    if (first_student not in graph) or (second_student not in graph):
        print("cannot find given student in the graph")
        return

    #create queue
    queue = deque([]) 
    #since it is a graph, we create visited list
    visited = [] 
    #create root node
    
    root = Node(first_student)
    
    #show the search tree explored so far
    treeplot = TreePlot()
    treeplot.generateDiagram(root, root)
    
    #add to queue and visited list
    queue.append(root)    
    visited.append(root.state)
    # check if there is something in queue to dequeue
    while len(queue) > 0:
        
        #get first item in queue
        currentNode = queue.popleft()
        
        #remove node from the frontier
        currentNode.fringe = False
        
        #print (("-- dequeue --"), currentNode.state)
        
        #check if this is goal student
        if currentNode.state == second_student:
            print ("students connected")
            #print the path
            print ("----------------------")
            print ("Path")
            currentNode.printPath()
            
            #show the search tree explored so far
            treeplot = TreePlot()
            treeplot.generateDiagram(root, currentNode)
            break
        
        
        #get the child nodes 
        childStates = graph[currentNode.state]
     
        for childState in childStates:
            
            childNode = Node(childState)
            
            #check if node is not visited
            if childNode.state not in visited:
                
                #add this node to visited nodes
                visited.append(childNode.state)
                
                
                #add to tree and queue
                currentNode.addChild(childNode)
                queue.append(childNode)    

        #show the search tree explored so far
        treeplot = TreePlot()
        treeplot.generateDiagram(root, currentNode)
                    
    #print tree
    print ("----------------------")
    print ("Tree")
    root.printTree()
    
    
    # if relationship cannot be established print a message
    if second_student not in visited:
        print()
        print("relationship cannot be established")
    


first_student="Dolly"
second_student="Liangyu"
BFS_Liangyu(graph, first_student, second_student)

time.sleep(5)


first_student="George"
second_student="Bob"
BFS_Liangyu(graph, first_student, second_student)


