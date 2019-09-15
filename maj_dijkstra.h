#ifndef MAJ_DIJKSTRA_H
#define MAJ_DIJKSTRA_H
#define key_type distance_type

#include "maj_implicit_heap.h"




static void single_dijkstra_iteration(implicit_heap *heap, int *indicies, distance_type *edgeWeights, int *nnCounts, distance_type *distances, unsigned int dstride){
    
    unsigned int i;
    
    distance_type minDist=heap->root[0].key;
    int minPosition=heap->root[0].originalIndex;
    //distances[minPosition*dstride]=minDist;
    for(i=nnCounts[minPosition];i<nnCounts[minPosition+1];i++){
        unsigned int index=indicies[i];
        distance_type eWeight=edgeWeights[i];
        distance_type current=distances[index*dstride];
        distance_type possible=eWeight+minDist;
        if(possible<current){
            distances[index*dstride]=possible;
            unsigned int location=heap->locations[index];
            decrease_key(heap,location,possible);
        }
        
    }
    delete_min(heap);
    
}




static void dijkstra_distances(int *indicies, distance_type *edgeWeights, int *nnCounts, distance_type *distances,  int pcount,  int dstride){
    
    implicit_heap heap=create_heap_with_batch(distances,pcount, dstride);
    
    
    unsigned int iter=0;
    
    
    // printf("added\n");
    while(!empty(&heap)){
        
        
        single_dijkstra_iteration(&heap, indicies, edgeWeights, nnCounts, distances, dstride);
        iter++;
        
    }
    
    
    free(heap.root);
    free(heap.locations);

}

static void labeled_single_dijkstra_iteration(implicit_heap *heap, int *indicies, distance_type *edgeWeights, int *nnCounts, distance_type *distances, int *labels, unsigned int dstride){
    
    unsigned int i;
    
    distance_type minDist=heap->root[0].key;
    int minPosition=heap->root[0].originalIndex;
    //distances[minPosition*dstride]=minDist;
    for(i=nnCounts[minPosition];i<nnCounts[minPosition+1];i++){
        unsigned int index=indicies[i];
        distance_type eWeight=edgeWeights[i];
        distance_type current=distances[index*dstride];
        distance_type possible=eWeight+minDist;
        if(possible<current){
            distances[index*dstride]=possible;
            labels[index]=labels[minPosition];
            if(current==FLT_MAX){
                insert_node(heap, index, possible);
            }else{
                unsigned int location=heap->locations[index];
                decrease_key(heap,location,possible);
            }
           
        }
        
    }
    delete_min(heap);
    
}




static void labeled_dijkstra(int *indicies, distance_type *edgeWeights, int *nnCounts, distance_type *distances, int *labels, int pcount,  int dstride){
    
   int i;
    //int sq=sqrt(pcount);
    //implicit_heap heap=create_heap_with_batch(distances,pcount, dstride);
    implicit_heap heap=create_empty_heap_with_locations(pcount);
    for(i=0;i<pcount;i++){
        if(distances[i*dstride]!=FLT_MAX){
            insert_node(&heap,i,distances[i*dstride]);
        }
    }
    
    unsigned int iter=0;
    
    
    // printf("added\n");
    while(!empty(&heap)){
        
        
        labeled_single_dijkstra_iteration(&heap, indicies, edgeWeights, nnCounts, distances, labels, dstride);
        iter++;
        
    }
    
    
    free(heap.root);
    free(heap.locations);
    
}

#endif






