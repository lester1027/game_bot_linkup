"""
All coordinates assume a screen resolution of 1920x1200, and Chrome 
maximized with the Bookmarks Toolbar enabled.
"""


from PIL import ImageGrab
import time
import pyautogui
import numpy as np
import cv2
from keras.models import load_model

from utilities.preprocessing.simple_preprocessor import SimplePreprocessor
from utilities.preprocessing.imagetoarray_preprocessor import ImageToArrayPreprocessor

from sklearn.preprocessing import MinMaxScaler

n_col=12
n_row=8

#top_left x and y of the top-left box
x_ini=579
y_ini=500

x_end=1315
y_end=990

scaler = MinMaxScaler()
#resize pixel
pixels=28


pos_dict={}


for row in range(n_row):
    for col in range(n_col):
        pos_dict[(row,col)]=(int(x_ini+(x_end-x_ini)/n_col*col),int(y_ini+(y_end-y_ini)/n_row*row),
                int(x_ini+(x_end-x_ini)/n_col*(col+1)),int(y_ini+(y_end-y_ini)/n_row*(row+1)))

cen_dict={}

for row in range(n_row):
    for col in range(n_col):
        cen_dict[(row,col)]=((int((pos_dict[(row,col)][0]+pos_dict[(row,col)][2])/2)),
                int((pos_dict[(row,col)][1]+pos_dict[(row,col)][3])/2))

type_list=['agent', 'bear', 'cat', 'dog', 'dragon', 'duck',
       'elephant', 'geisha', 'girl', 'horse', 'mask',
       'octopus', 'others','rabbit', 'shell', 'spike', 'tank']
    
def grabBox():
    '''
    #input:None
    #output: a dictionary containing the cropped and grayscaled images
    '''
    val_dict={}
    #capture the fullscreen
    im =ImageGrab.grab()
    im=np.array(im)
    #grayscaling for lower computational cost
    im_gray= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('full.png',im_gray)
    #size scaling
    im_gray=im_gray.ravel().reshape(-1,1)
    #rescale the values from 0 to 1
    #if directly opened with an image explorer, black image
    im_scale=scaler.fit_transform(im_gray)
    im_scale=im_scale.reshape(-1,1200,1920,1)
    for row in range(n_row):
        for col in range(n_col):
            box=pos_dict[(row,col)]
            im_crop=im_scale[0][box[1]:box[3],box[0]:box[2],:]
            val_dict[(row,col)]=im_crop
            #cv2.imwrite('({},{})'.format(row,col)+'.png', im_crop)
    return val_dict

#one mouse left click
def leftClick():
    pyautogui.click()
    #completely optional. But nice for debugging purposes.
    #print("Click.")

#move the cursor to the input coordinates
def mousePos(cord):
    pyautogui.moveTo((cord[0], cord[1]))

def loadModel():
    #load the trained CNN model
    model=load_model('output/weights-004-0.0017.hdf5')
    return model


def categorize(model):
    '''
    #input:None
    #output: a dictionary containing the cropped and grayscaled images
    '''
    val_dict =grabBox()
    type_dict={}
    

    #for each position of the game panel
    for row in range(n_row):
        for col in range(n_col):
            val=val_dict[(row,col)]
            #resize
            sp = SimplePreprocessor(pixels,pixels)
            val=sp.preprocess(val)
            #image to array operation
            iap = ImageToArrayPreprocessor()
            val=iap.preprocess(val)
            val=val.reshape(-1,pixels,pixels,1)
            type_dict[(row,col)]=type_list[model.predict(val,batch_size=1).argmax()]
    return type_dict

def fill_type(type_dict):
    ''' 
    add the outer layer of the game panel to the type_dict
    all of them belong to the type 'others'
    '''
    for i in range(-1,13):
        type_dict[(-1,i)]='others'
        type_dict[(8,i)]='others'
        
    for j in range(0,8):
        type_dict[(j,-1)]='others'
        type_dict[(j,12)]='others'
    
    return type_dict

def create_graph(type_dict,desired_box='octopus'):
    '''
    input: select a type of box to form a graph
    output: coord_graph,coord_desired,coord_others,coord_obstacle
    '''
    #creat a grid graph dict, stating all the connections of adjacent grids
    coord_graph={}
    for i in list(type_dict.keys()):
        coord_graph[i]={}
        for j in list(type_dict.keys()):
            #if i and j boxes are adjacent to each other
            if (abs(i[0]-j[0])==1 and abs(i[1]-j[1])==0) or (abs(i[1]-j[1])==1 and abs(i[0]-j[0])==0):
                coord_graph[i][j]=1
    
    #a list containing the coordinates of the desired box
    coord_desired=[]
    #a list containing the coordinates of other boxes
    coord_others=[]
    for coord, box_type in list(type_dict.items()):
        if box_type==desired_box:
            coord_desired.append(coord)
        if box_type=='others':
            coord_others.append(coord)
    
    #coordinates of obstacles
    coord_obstacle=list(set(type_dict.keys())-set(coord_desired)-set(coord_others))
    
    #for each pair of coordinates apart from those of desired boxes
    for k in coord_obstacle:
        coord_graph.pop(k)
        
    for m in list(coord_graph.keys()):
        #for each pair of obstacle coordinates
        for n in coord_obstacle:
            #if the coord_obstacle is included in the neighbor points
            if n in coord_graph[m].keys():
                #remove the coodinates
                coord_graph[m].pop(n)
    
    return coord_graph,coord_desired,coord_others,coord_obstacle



def FIFO_push(FIFO_list,element):
    return FIFO_list.append(element)

def FIFO_pop(FIFO_list):
    return FIFO_list.pop(0)

def BFS(maze_graph, initial_vertex) :
    '''
     input: the maze map that is a graph represented in a dictionary
     output: a list with the order of executed vertices and a dictionary 
             containing the vertices as the keys and its parents as the value
     '''
    # explored vertices list
    explored_vertices = list()
    
    #FIFO stack
    queuing_structure = list()
    
    #Parent Dictionary
    parent_dict = dict()
        

    FIFO_push(queuing_structure,(initial_vertex,None)) # push the initial vertex to the queuing_structure
    while len(queuing_structure) > 0: #   while queuing_structure is not empty:
        current_vertex,parent = queuing_structure.pop(0)
        
        #determine of which direction the box was being pushed to the queue
        if parent==None:
            push_dir=[]     
        elif abs(parent[0]-current_vertex[0])==0:
            push_dir='x'
        elif abs(parent[1]-current_vertex[1])==0:
            push_dir='y'
            
        # if the current vertex is not explored
        if current_vertex not in explored_vertices:
            # add current_vertex to explored vertices
            explored_vertices.append(current_vertex)
            # use parent_dict to map the parent of the current vertex
            parent_dict[current_vertex]=parent
            
            def same_dir(key):
                #key function to determine of which direction the box 
                #is being pushed to the queue
                if abs(key[0]-current_vertex[0])==0:
                    push_dir_new='x'
                elif abs(key[1]-current_vertex[1])==0:
                    push_dir_new='y'
                if push_dir_new==push_dir:
                    return 'a'
                else:
                    return 'b'
            #sort the neighbor points such that the one requires no turn
            #can be pushed first
            neighbor_list=sorted(maze_graph[current_vertex].keys(),key=same_dir)
            # for each neighbor of the current vertex in the maze graph:
            for neighbor in neighbor_list:
                # if neighbor is not explored:
                if neighbor not in explored_vertices:

                    # push the tuple (neighbor,current_vertex) to the queuing_structure
                    FIFO_push(queuing_structure,(neighbor,current_vertex))
                    
    return explored_vertices,parent_dict

def create_walk_from_parents(parent_dict,initial_vertex,target_vertex):
    #initiate the walk
    walk=[]
    parent=target_vertex
    
    if parent==None:
        return []
    
    else:
        walk.append(target_vertex)
        while parent!=initial_vertex and parent!=None:
            parent=parent_dict[parent]
            walk.append(parent)

        walk.pop(-1)
        return walk[::-1]


def count_turn(walk):
    moving_coord=[]
    #for each step in the walk, determine the direction of movement
    for w in range(len(walk)-1):
        coord1=walk[w]
        coord2=walk[w+1]
        if abs(coord1[0]-coord2[0])==1 and abs(coord1[1]-coord2[1])==0:
            moving_coord.append('y')
        elif abs(coord1[1]-coord2[1])==1 and abs(coord1[0]-coord2[0])==0:
            moving_coord.append('x')
    
    #count the number of turning times
    turn_count=0
    for m in range(len(moving_coord)-1):
        if moving_coord[m] != moving_coord[m+1]:
            turn_count+=1
    
    return turn_count



#%%

'''
def main():
    #load model
    model=loadModel()
    while True:
        type_dict=categorize(model)
        for i in list(type_dict.keys()):
                type1=type_dict[(i)]
                for j in list(type_dict.keys()):
                        type2=type_dict[(j)]
                        if type1==type2:
                            #move to the first desired box
                            mousePos(cen_dict[(i)])
                            leftClick()
                            #move to the second desired box
                            mousePos(cen_dict[(j)])
                            leftClick()
         
'''                    

def main():
    #load model
    model=loadModel()

    while True:
        type_dict=categorize(model=model)
        #print('here1')
        #create a graph by randomly choose a desired box (exclude 'others')
        desired_box_list=list(set(type_dict.values())-set(['others']))
        #for each desire box
        for desired_box in desired_box_list:
            #print('here2')
            coord_graph,coord_desired,coord_others,coord_obstacle=create_graph(type_dict=type_dict,desired_box=desired_box)
            for coord1 in coord_desired[:-1]:
                initial_vertex=coord1
                #generate the routing table
                explored_vertices,parent_dict=BFS(coord_graph,initial_vertex)
                #if the chosen initial vertex is not isolated
                if list(parent_dict.values())!=[None]:
                
                    for coord2 in coord_desired[(coord_desired.index(coord1)+1):]:
                        target_vertex=coord2

                        #if the target_vertex is reachable by checking the parent_dict
                        if target_vertex in list(parent_dict.keys()):
                            #generate the walk
                            walk=create_walk_from_parents(parent_dict,initial_vertex,target_vertex)
                            turn_count=count_turn(walk)
                        
                        
                            if turn_count<=2:
                                #move to the first desired box
                                mousePos(cen_dict[initial_vertex])
                                leftClick()
                                time.sleep(0.1)
                                #move to the second desired box
                                mousePos(cen_dict[target_vertex])
                                leftClick()
                                break
                                
                            else:
                                #check next target_vertex
                                continue
                            
                        else:
                            #check next target_vertex
                            continue
                    break
                else:
                    #check next initial_vertex
                    continue
                     
                   
        
if __name__ == '__main__':
    main()
