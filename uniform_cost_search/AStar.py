'''
@author: Devangini
'''

'''
Liangyu Wang
980025288
'''

from State import State
from Node import Node
import queue
from TreePlot import TreePlot
    

def performUCS():
    """
    This method performs UniformCostSearch search
    """
    
    #create queue
    pqueue = queue.PriorityQueue()
    
    #create visited nodes list
    visited=[]

    #create root node
    ecount=0
    initialState = State()
    root = Node(initialState, None)
    

    
    #show the search tree explored so far
    treeplot = TreePlot()
    treeplot.generateDiagram(root, root)
    
    #add to priority queue
    pqueue.put((root.costFromRoot, ecount, root))
    

    # add to visited list
    visited.append(root.state.place)
    
    #check if there is something in priority queue to dequeue
    while not pqueue.empty(): 
        
        #dequeue nodes from the priority Queue
        _, _, currentNode = pqueue.get()
        
        
        #remove from the fringe
        currentNode.fringe = False
        
        #check if it has goal State
        print ("-- dequeue --", currentNode.state.place)
        
        #check if this is goal state
        if currentNode.state.checkGoalState():
            print ("reached goal state")
            #print the path
            print ("----------------------")
            print ("Path")
            currentNode.printPath()
            
            #show the search tree explored so far
            treeplot = TreePlot()
            treeplot.generateDiagram(root, currentNode)
            break
            
        #get the child nodes 
        childStates = currentNode.state.successorFunction()
        for childState in childStates:
            
            childNode = Node(State(childState), currentNode)
            
            #check if node is in visited list
            if childNode.state.place not in visited:
                visited.append(childNode.state.place)
                ecount+=1
                #add to tree and queue
                pqueue.put((childNode.costFromRoot, ecount, childNode))
            
        #show the search tree explored so far
        treeplot = TreePlot()
        treeplot.generateDiagram(root, currentNode)
        
                
    #print tree
    print ("----------------------")
    print ("Tree")
    root.printTree()
    
performUCS()