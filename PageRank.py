
#   Author: Sean Saathoff
#   Goal: Implement PageRank algorithm and query-based topic sensitive
#       PageRank;


import ir470
import os
import numpy as np

# ########################################
# now, write some code
# ########################################

# The class to implement PageRank algorithm and 
#   Query-Based topic sensitive PageRank
class PageRank(object):
    def __init__(self):
        pass
    
    #   This function takes the filename of the matrix, and returns 
    #   the transistion matrix in an array from using numpy
    def create_link_matrix(self, transition_matrix_filename):
        f = open(transition_matrix_filename, 'rU')
        num_rows = 0
        for line in f:
            line = line.split()
            if int(line[0]) > num_rows:     #count number of nodes in graph
                num_rows = int(line[0])
            elif int(line[1]) > num_rows:
                num_rows = int(line[1])
    
        link_matrix = np.zeros(shape=(num_rows, num_rows))  #create graph
        f.close()
        f = open(transition_matrix_filename, 'rU')
        for line in f:
            line = line.split()
            link_matrix.itemset((int(line[0])-1,int(line[1])-1), int(line[2]))  #insert edges
        row_count = 0
        for row in link_matrix:
            column_count = 0;
            row_sum = sum(row)
            if row_sum != 0:
                for item in row:            #normalize each row
                    item = item/row_sum
                    link_matrix.itemset(row_count,column_count, item)
                    column_count += 1
            elif row_sum == 0:
                for item in row:
                    item = 1/num_rows
            row_count += 1
        
        return link_matrix              #return link matrix
    
    # cal_pagerank( self, transition_matrix_filename )
    # purpose:
    #   This function takes the filename of the transition matrix as input,
    #   and calculate the PageRank scores for each document using Power Method
    # returns: a list of pairs of document_id, and its PageRank score, sorted
    #   by the descending order of the PageRank score.
    # parameters:
    #   transition_matrix_filename - the file name for transition matrix
    def cal_pagerank(self, transition_matrix_filename):
        
        pagerank_scores = []
             
        link_matrix = self.create_link_matrix(transition_matrix_filename)
        
        num_nodes = link_matrix.shape[0]            #create teleport matrix
        tele_matrix = np.zeros(shape=(num_nodes, num_nodes))
        tele_matrix.fill(1.0/num_nodes)     
        
        link_matrix = .9*link_matrix            #multiply by 1-alpha
        tele_matrix = .1*tele_matrix
        
        trans_prob_matrix = link_matrix + tele_matrix       #get final transistion matrix
        
        vector_mat = np.zeros(shape=(1, num_nodes)) #initial vector 
        vector_mat.itemset(0, 0, 1)
                          
        count = 0;
        while True:     #loop through until vector converges
            count += 1
            next_vect = np.dot(vector_mat,trans_prob_matrix)
            if (np.around(next_vect, decimals = 5) == np.around(vector_mat, decimals = 5)).all():   #check convergence to 3 decimal points
                break
            vector_mat = next_vect
        
        count = 1
        for item in next_vect[0]:
                pagerank_scores.append((count, item))
                count += 1
        pagerank_scores = sorted(pagerank_scores, key=lambda x: x[1], reverse = True)
       
        return pagerank_scores

    # cal_tspr( self, transition_matrix_filename, doc_topics_filename )
    # purpose:
    #   This function takes the filename of the transition matrix, and
    #   the filename of the topics for each document as input,
    #   and calculate the topic sensitive PageRank scores for each document
    #   using Power Method
    # returns: a dictionary which includes pairs of topic ID, and its
    #   corresponding topic sensitive PageRank scores.
    # parameters:
    #   transition_matrix_filename - the file name for transition matrix
    #   doc_topics_filename - the file name for the topics for each document
    def cal_tspr(self, transition_matrix_filename, doc_topics_filename):
        tspr_dict = {}
        node_top_dict = {}      #stores topics and their corresponding nodes
        f = open(doc_topics_filename, 'rU')
        for line in f:
            temp_list = line.split()            #get topic name
            if temp_list[1] in node_top_dict:     #if topic in dictionary, add corresponding node  
                node_top_dict[temp_list[1]].append(temp_list[0])
            else:       #otherwise add topic to dictionary
                node_top_dict[temp_list[1]] = [temp_list[0]]
                    
        link_matrix = self.create_link_matrix(transition_matrix_filename)       #create same link matrix
        link_matrix = .9*link_matrix            #multiply by 1-alpha
                    
        num_nodes = link_matrix.shape[0]            #create teleport matrix
        tele_matrix = np.zeros(shape=(num_nodes, num_nodes))
        vector_mat = np.zeros(shape=(1, num_nodes)) #initial vector 
        
        for key in node_top_dict:           #do separate pagerank for each topic
            for node in node_top_dict[key]:     #set transistion matrix
                for i in range(0,num_nodes):
                    tele_matrix.itemset(i, int(node)-1, 1.0/len(node_top_dict[key]))
            
            tele_matrix = .1 * tele_matrix
    
            trans_prob_matrix = link_matrix + tele_matrix       #get final transistion matrix
    
            vector_mat.fill(0)
            vector_mat.itemset(0, 0, 1)
    
            count = 0;
                
            while True:     #loop through until vector converges
                count += 1
                next_vect = np.dot(vector_mat,trans_prob_matrix)
                if (np.around(next_vect, decimals = 5) == np.around(vector_mat, decimals = 5)).all():   #check convergence to 3 decimal points
                    break
                vector_mat = next_vect
    
            count = 1
            pagerank_scores = []
            for item in next_vect[0]:
                pagerank_scores.append((count, item))
                count += 1
            pagerank_scores = sorted(pagerank_scores, key=lambda x: x[1], reverse = True)
            tspr_dict[key] = pagerank_scores        #set page rank scores for the topic

            tele_matrix.fill(0)                     #reset the tele matrix
        
        return tspr_dict 

    # cal_qtspr( self, tspr_dict, query )
    # purpose:
    #   This function takes the dictionary of topic sensitive PageRank scores 
    #   and the query of topical distribution as input, and returns the
    #   corresponding query-based topic sensitive PageRank scores.
    #   and calculate the topic sensitive PageRank scores for each document
    #   using Power Method
    # returns: a list of pairs of document_id, and its qtspr score, sorted
    #   by the descending order of the qtspr score.
    # parameters:
    #   tspr_dict - the dictionary of tspr scores 
    #   query - the topical distribution for the query 
    def cal_qtspr(self, tspr_dict, query):
        qtspr_scores= []

        query_list = []
        query_list = query.split()
        query_dict = {}
        for item in query_list:     #create dictionary key = topic value = probability
            temp_list = item.split(':')
            query_dict[temp_list[0]] = float(temp_list[1])
        
        qtspr_scores_dict = {}          #used to hold qtspr scores corresponding to nodes
        num_nodes = len(tspr_dict.itervalues().next())
        
        for i in range(1,num_nodes+1):      #set all scores to 0
            qtspr_scores_dict[i] = 0
                
        for key in tspr_dict:
            prob = query_dict[key]      #store probability value for the topic
            temp_list = tspr_dict[key]
            for tuple in temp_list:         #update qtspr score based on the topic
                qtspr_scores_dict[tuple[0]] += tuple[1]*prob
                
        for key in qtspr_scores_dict:       #create and sort list of tuples 
            qtspr_scores.append((key, qtspr_scores_dict[key]))
        
        qtspr_scores = sorted(qtspr_scores, key=lambda x: x[1], reverse = True)
        
        return qtspr_scores

# now, we'll define our main function which actually starts the indexer and
# does a few queries
def main(args):
    
    print "instantiate PageRank class"
    pagerank = PageRank()
    print "calculating PageRank scores"
    pagerank_scores = pagerank.cal_pagerank('example_transition_matrix.txt')
    print "PageRank scores: ", pagerank_scores
    

    # The code to write the pagerank scores to a text file
    output_filename = 'hw3_pagerank.txt'
    output_file = open(output_filename, 'w')
    output_str = ""
    for doc_id, score in pagerank_scores:
        output_str += "%s:%s\t" %(doc_id, score)
    output_str = output_str.strip()
    output_file.write("%s" % (output_str))
    output_file.close()



    # The code to write the qtspr scores to a text file
    output_filename = 'hw3_qtspr.txt'
    output_file = open(output_filename, 'w')

    print "calculating Topic Sensitive PageRank scores"
    tspr_dict = pagerank.cal_tspr('example_transition_matrix.txt', \
        'example_doc_topics.txt')

    for query in ('sports:0.3 finance:0.2 technology:0.4 politics:0.1', \
        'sports:0.3 finance:0.3 politics:0.3 technology:0.1'):
        
        qtspr_scores = pagerank.cal_qtspr(tspr_dict, query)
        print "Query-based Sensitive PageRank Scores for query '%s' are: %s" \
            % (query, qtspr_scores)
        
        output_str = ""
        for doc_id, score in qtspr_scores:
            output_str += "%s:%s\t" %(doc_id, score)
        output_str = output_str.strip()
        output_file.write("%s\n" % (output_str))
    
    output_file.close()

# this little helper will call main() if this file is executed from the command
# line but not call main() if this file is included as a module
if __name__ == "__main__":
    import sys
    main(sys.argv)

