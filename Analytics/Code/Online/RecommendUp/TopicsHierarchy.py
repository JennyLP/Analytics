import networkx as nx
import numpy as np
import math
import bz2
import matplotlib.pyplot as plt
import itertools

from networkx import single_source_shortest_path, single_source_shortest_path_length
#from networkx.search import dfs_postorder,dfs_preorder


def dfs_preorder(G,v=None):
    """
    Return a list of nodes ordered by depth first search (DFS) preorder.
    If the graph has more than one component return a list of lists.
    Optional v=v limits search to component of graph containing v.
    """
    V=SC.Preorder(G, queue=networkx.queues.DFS)
    V.search(v)
    return V.forest


def dfs_postorder(G,v=None):
    """
    Return a list of nodes ordered by depth first search (DFS) postorder.
    If the graph has more than one component return a list of lists.
    Optional v=v limits search to component of graph containing v.
    """
    V=SC.Postorder(G,queue=networkx.queues.DFS)
    V.search(v)
    return V.forest


def dfs_predecessor(G,v=None):
    """
    Return a dictionary of nodes each with a list of predecessor
    nodes in depth first search (DFS) order.
    Optional v=v limits search to component of graph containing v.
    """
    V=SC.Predecessor(G,queue=networkx.queues.DFS)
    V.search(v)
    return V.data


def dfs_successor(G,v=None):
    """
    Return a dictionary of nodes each with a list of successor
    nodes in depth first search (DFS) order.
    Optional v=v limits search to component of graph containing v.
    """
    V=SC.Successor(G,queue=networkx.queues.DFS)
    V.search(v)
    return V.data

def bfs_length(G,source,target=None):
    """
    Return a dictionary of nodes with the shortest path length from source.
    """
    V=SC.Length(G,queue=networkx.queues.BFS)
    V.search(source)
    if target!=None:
        try:
            return V.length[target]
        except KeyError:
            return -1 # no target in graph
    else:
        return V.length

def bfs_path(G,source,target=None):
    """
    Return a dictionary of nodes with the paths
    from source to all reachable nodes.
    Optional target=target produces only one path as a list.
    """
    V=SC.Predecessor(G,queue=networkx.queues.BFS)
    V.search(source)
    if target!=None:
        path=V.path(target)
        path.insert(0,source)
        return path # return one path
    else:
        paths={}
        for k in V.data.keys():
            paths[k]=V.path(k)
            paths[k].insert(0,source)
        return paths


def dfs_forest(G,v=None):
    """
    Return a forest of trees built from depth first search (DFS).
    Optional v=v limits search to component of graph containing v
    and will return a single tree.
    """
    V=SC.Forest(G,queue=networkx.queues.DFS)
    V.search(v)
    return V.forest


def connected_components(G):
    """
    Return a list of lists of nodes in each connected component of G.

    The list is ordered from largest connected component to smallest.
    For undirected graphs only. 
    """
    if G.is_directed():
        raise networkx.NetworkXError,\
              """Not allowed for directed graph G.
              Use UG=G.to_undirected() to create an undirected graph."""
    seen={}
    components=[]
    for v in G:      
        if v not in seen:
            c = single_source_shortest_path_length(G,v)
            components.append(c.keys())
            seen.update(c)
    components.sort(lambda x, y: cmp(len(y),len(x)))
    return components            


def number_connected_components(G):
    """Return the number of connected components in G.
    For undirected graphs only. 
    """
    return len(connected_components(G))


def is_connected(G):
    """Return True if G is connected.
    For undirected graphs only. 
    """
    if G.is_directed():
        raise networkx.NetworkXError,\
              """Not allowed for directed graph G.
              Use UG=G.to_undirected() to create an undirected graph."""
    return len(single_source_shortest_path(G, G.nodes_iter().next()))==len(G)


def connected_component_subgraphs(G):
    """
    Return a list of graphs of each connected component of G.
    The list is ordered from largest connected component to smallest.
    For undirected graphs only. 

    For example, to get the largest connected component:
    >>> H=connected_component_subgraphs(G)[0]

    """
    cc=connected_components(G)
    graph_list=[]
    for c in cc:
        graph_list.append(G.subgraph(c,inplace=False))
    return graph_list


def node_connected_component(G,n):
    """
    Return a list of nodes of the connected component containing node n.

    For undirected graphs only. 

    """
    if G.is_directed():
        raise networkx.NetworkXError,\
              """Not allowed for directed graph G.
              Use UG=G.to_undirected() to create an undirected graph."""
    return single_source_shortest_path_length(G,n).keys()



def strongly_connected_components(G):
    """Returns list of strongly connected components in G.
     Uses Tarjan's algorithm with Nuutila's modifications.
     Nonrecursive version of algorithm.

     References:

      R. Tarjan (1972). Depth-first search and linear graph algorithms.
      SIAM Journal of Computing 1(2):146-160.

      E. Nuutila and E. Soisalon-Soinen (1994).
      On finding the strongly connected components in a directed graph.
      Information Processing Letters 49(1): 9-14.

     """
    neighbors=G.neighbors
    preorder={}
    lowlink={}    
    scc_found={}
    scc_queue = []
    scc_list=[]
    i=0     # Preorder counter
    for source in G:
        if source not in scc_found:
            queue=[source]
            while queue:
                v=queue[-1]
                if v not in preorder:
                    i=i+1
                    preorder[v]=i
                done=1
                for w in neighbors(v):
                    if w not in preorder:
                        queue.append(w)
                        done=0
                        break
                if done==1:
                    lowlink[v]=preorder[v]
                    for w in neighbors(v):
                        if w not in scc_found:
                            if preorder[w]>preorder[v]:
                                lowlink[v]=min([lowlink[v],lowlink[w]])
                            else:
                                lowlink[v]=min([lowlink[v],preorder[w]])
                    queue.pop()
                    if lowlink[v]==preorder[v]:
                        scc_found[v]=True
                        scc=[v]
                        while scc_queue and preorder[scc_queue[-1]]>preorder[v]:
                            k=scc_queue.pop()
                            scc_found[k]=True
                            scc.append(k)
                        scc_list.append(scc)
                    else:
                        scc_queue.append(v)
    scc_list.sort(lambda x, y: cmp(len(y),len(x)))
    return scc_list


def kosaraju_strongly_connected_components(G, source=None):
    """Returns list of strongly connected components in G.
     Uses Kosaraju's algorithm.
     """
    components = []
    post = dfs_postorder(G,source=source, reverse_graph=True)
    seen = {}
    while post:
        r = post.pop()
        if r in seen:
            continue
        c = dfs_preorder(G,r)
        new=[v for v in c if v not in seen]
        seen.update([(u,True) for u in new])
        components.append(new)
    components.sort(lambda x, y: cmp(len(y),len(x)))
    return components            


def strongly_connected_components_recursive(G):
    """Returns list of strongly connected components in G.
     Uses Tarjan's algorithm with Nuutila's modifications.
     this recursive version of the algorithm will hit the
     Python stack limit for large graphs.
     
     """
    def visit(v,cnt):
        root[v] = cnt
        visited[v] = cnt
        cnt += 1
        stack.append(v)
        for w in G[v]:
            if w not in visited: visit(w,cnt)
            if w not in component:
                root[v] = min(root[v],root[w])
        if root[v] == visited[v]:
            component[v] = root[v]
            tmpc=[v] # hold nodes in this component
            while stack[-1] != v:
                w=stack.pop()                
                component[w]=root[v]
                tmpc.append(w)
            stack.remove(v) 
            scc.append(tmpc) # add to scc list
    scc = []
    visited = {}   
    component = {}
    root = {}
    cnt = 0
    stack = []
    for source in G:
        if source not in visited: 
            visit(source,cnt)

    scc.sort(lambda x, y: cmp(len(y),len(x)))
    return scc


def strongly_connected_component_subgraphs(G):
    """
    Return a list of graphs of each strongly connected component of G.
    The list is ordered from largest connected component to smallest.

    For example, to get the largest strongly connected component:
    >>> H=strongly_connected_component_subgraphs(G)[0]

    """
    cc = strongly_connected_components(G)
    graph_list = []
    for c in cc:
        graph_list.append(G.subgraph(c,inplace=False))
    return graph_list


def number_strongly_connected_components(G):
    """Return the number of connected components in G.
    For undirected graphs only. 
    """
    return len(strongly_connected_components(G))


def is_strongly_connected(G):
    """Return True if G is strongly connected.
    """
    if not G.is_directed():
        raise networkx.NetworkXError,\
              """Not allowed for undirected graph G.
              See is_connected() for connectivity test."""
    return len(strongly_connected_components(G)[0]) == len(G)


def buildNetworkFromList_old( usersPreferences ):

    myTopicsGraph = nx.Graph()

    for userId, userPreferences in usersPreferences.iteritems():

        for topic1 in userPreferences:
            for topic2 in userPreferences:

                if topic1 == topic2:
                    continue

                if myTopicsGraph.has_edge(topic1, topic2):
                    # edge already exists, increasing weight...
                    myTopicsGraph[topic1][topic2]['weight'] += 0.5
                else:
                    # new edge. add with weight=1
                    myTopicsGraph.add_edge(topic1, topic2, weight=0.5)

    #print "-->", myTopicsGraph.edges(data=True)
    return myTopicsGraph

    #nx.draw( myTopicsGraph, with_labels=True )
    #nx.draw( myTopicsGraph, pos = nx.spectral_layout( myTopicsGraph ), nodecolor='r', edge_color='b', with_labels=True )

    pos=nx.spring_layout(myTopicsGraph) # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(myTopicsGraph, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(myTopicsGraph, pos, width=6)
    nx.draw_networkx_edges(myTopicsGraph,pos,edgelist=esmall, width=6,alpha=0.5,edge_color='b',style='dashed')

    # labels
    nx.draw_networkx_labels(myTopicsGraph,pos,font_size=20,font_family='sans-serif')

    plt.axis('off')
    #plt.savefig("weighted_graph.png") # save as png
    plt.show() # display


def fancyTopicClassification( usersPreferences ):

    myUsersTopics = usersPreferences.values()
    # flattening list of list
    myTopics = list( itertools.chain(*usersPreferences.values()) )
    # make sure no keyword is used multiple times for same user
    myTopics = list ( set(myTopics) )
    print myTopics

    myAdjacencyMatrix = [ [ 0.0 for t2 in myTopics ] for t1 in myTopics ]

    for topic1 in myTopics:
        for topic2 in myTopics:

            print topic1, topic2

            count1 = 0
            count2 = 0

            for myUserTopics in myUsersTopics:
                if topic1 in myUserTopics:
                    count1 += 1
                    if topic2 in myUserTopics:
                        count2 += 1

            if count1 != 0:
                myAdjacencyMatrix[ myTopics.index(topic1) ][ myTopics.index(topic2) ] = float( count2 ) / count1

    return myAdjacencyMatrix, myTopics


def computeFromAdjacency( myAdjacencyMatrix, myHeaders ):

    myAdjMatrix = {}

    for i in range( len(myAdjacencyMatrix) ):
        for j in range( len(myAdjacencyMatrix) ):
            myAdjMatrix[i] = { j : { 'weight' : myAdjacencyMatrix[i][j] } }

    G = nx.DiGraph(myAdjMatrix, format = 'weighted_adjacency_matrix')  # graph from matrix
    nx.draw(G, edge_labels = True, graph_border = True)
    plt.show()


def buildNetworkFromList( usersPreferences ):

    myGraph = nx.MultiDiGraph()
    game={}
    lines = (line.decode().rstrip('\r\n') for line in datafile)
    
    for line in lines:
        if line.startswith('['):
            tag,value=line[1:-1].split(' ',1)
            game[str(tag)]=value.strip('"')
        else:
        
            if game:
                white=game.pop('White')
                black=game.pop('Black')
                myGraph.add_edge(white, black, **game)
                game={}

    return myGraph


def plotFancyGraph( usersPreferences ):

    myGraph = buildNetworkFromList_old( usersPreferences )

    nsimilarities  = myGraph.number_of_edges()
    ntopics = myGraph.number_of_nodes()

    # identify connected components
    # of the undirected version

    #Gcc = list( nx.connected_component_subgraphs(myGraph.to_undirected()) )
    Gcc = strongly_connected_components_subgraphs( myGraph )

    if len(Gcc)>1:
        print("Strongly connected component consisting of:")
        print(Gcc[1].nodes())

    myGraph = buildNetworkFromList_old( usersPreferences )

    nsimilarities  = myGraph.number_of_edges()
    ntopics = myGraph.number_of_nodes()

    # identify connected components
    Gcc = list(nx.connected_component_subgraphs(myGraph.to_undirected()))

    if len(Gcc)>1:
        print("Note the disconnected component consisting of:")
        print(Gcc[1].nodes())

    # make new undirected graph H without multi-edges
    H = nx.Graph(myGraph)

    # edge width is proportional number of games played
    edgewidth = []
    for (u,v,d) in H.edges(data=True):
        edgewidth.append( d['weight'] )

    print edgewidth

    '''
    # node size is proportional to number of games won
    wins = dict.fromkeys( myGraph.nodes(),0.0 )

    for (u,v,d) in myGraph.edges():

        r = d['Result'].split('-')

        if r[0] == '1':
            wins[u] += 1.0
        elif r[0] == '1/2':
            wins[u] += 0.5
            wins[v] += 0.5
        else:
            wins[v] += 1.0
    '''

    try:
        pos = nx.graphviz_layout(H)
    except:
        pos = nx.spring_layout( H,iterations=20 )

    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8,8))
    nx.draw_networkx_edges(H, pos, alpha=0.3, width=edgewidth, edge_color='m')
    #nodesize = [ v.weight for v in H ]
    nx.draw_networkx_nodes( H, pos, node_size = 1, node_color='w', alpha=0.4 )
    nx.draw_networkx_edges( H, pos, node_size = 0, edge_color='k', alpha = 0.4, width = 1 )
    nx.draw_networkx_labels( H, pos, fontsize = 14 )
    
    font = {'fontname'   : 'Helvetica',
            'color'      : 'k',
            'fontweight' : 'bold',
            'fontsize'   : 14}

    # change font and write text (using data coordinates)
    font = {'fontname'   : 'Helvetica',
    'color'      : 'r',
    'fontweight' : 'bold',
    'fontsize'   : 14}

    '''
    plt.text(0.5, 0.97, "edge width = # games played",
             horizontalalignment='center',
             transform=plt.gca().transAxes)
    plt.text(0.5, 0.94,  "node size = # games won",
             horizontalalignment='center',
             transform=plt.gca().transAxes)
    '''

    plt.axis('off')
    plt.show() # display
