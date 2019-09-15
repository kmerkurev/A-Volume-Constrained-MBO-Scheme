#include "mbo_convolution.h"
#include <assert.h>
#include <time.h>
#include <assert.h>


//randomly shuffles an array of integers

void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = n - 1; i > 0; i--) {
            size_t j = (unsigned int) (drand48()*(i+1));
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
    
   
}


//reads a file and returns the bytes

void *getFileData(const char *fileName){
    
    FILE *f=fopen(fileName,"r");
    size_t fsize;
    fseek(f,0L,SEEK_END);
    fsize=ftell(f);
    fseek(f,0L,SEEK_SET);
    void *data=malloc(fsize);
    fread(data,1,fsize,f);
    fclose(f);
    return data;
}

int compare_int(const void *p, const void *q){
    int i1=*((int *) p);
    int i2=*((int *) q);
    
    if(i2>i1){
        return -1;
    }else if(i1>i2){
        return 1;
    }else{
        return 0;
    }
}

int compare_float(const void *p, const void *q){
    int i1=*((float *) p);
    int i2=*((float *) q);
    
    if(i2>i1){
        return -1;
    }else if(i1>i2){
        return 1;
    }else{
        return 0;
    }
}

int compare_indexed_floats(const void *p, const void *q){
    
    indexedFloat f1=*((indexedFloat *) p);
    indexedFloat f2=*((indexedFloat *) q);
    
    if(f2.dist>f1.dist){
        return -1;
    }else if(f1.dist>f2.dist){
        return 1;
    }else{
        return 0;
    }
    
}

void sort_indexed_floats(indexedFloat *distances, int num){
    
    qsort(distances, num, sizeof(indexedFloat), &compare_indexed_floats);
    
}
void sort_int(int *array, int n){
    qsort(array,n,sizeof(int),&compare_int);
}
void sort_float(float *array, int n){
    qsort(array,n,sizeof(float),&compare_float);
}

indexedFloat *readAndConvertDistanceDataToIndexedFloat(const char *distanceFilename, const char *knnFilename, int maxNeighbors, int pcount){
    int i,j;
    float *distances=getFileData(distanceFilename);
    int *indicies=getFileData(knnFilename);
    indexedFloat *comboData=calloc(maxNeighbors*pcount,sizeof(indexedFloat));
    for(i=0;i<pcount;i++){
        
        for(j=0;j<maxNeighbors;j++){
            comboData[i*maxNeighbors+j].index=indicies[i*maxNeighbors+j];
            comboData[i*maxNeighbors+j].dist=distances[i*maxNeighbors+j];
            

        }
        sort_indexed_floats(&comboData[i*maxNeighbors],maxNeighbors);
        
    }
    free(distances);
    free(indicies);
    return comboData;
    
}


float kernel(float t){
    
    return exp(-t);
}



double factorial(int n){
    
    int i;
    double result=0;
    for(i=1;i<n;i++){
        result+=log(i+1);
    }
    
    return exp(result);
    
}


/* ----------------MODES-------------------
 
 
 
 Mode must be entered as a THREE letter code.  The first letter corresponds to the MBO method the second corresponds to the initialization method. The third letter determines whether fidelity nodes should be reweighted.
 
 
 METHOD CODES:
 k - convolution with symmetric weight matrix
 
 d - convolution with symmetric weight matrix squared
 
 INITIALIZATION CODES:
 
 r - Initializes the non-fixed labels with a random label.
 
 p - Initializes labels with the MBO algorithm
 
 v - Initializes the non-fixed labels by creating a voronoi diagram with the fixed labels as the seed points.  Every point is assigned the label of the fidelity point in its voronoi cell 
 
 
 REWEIGHT CODES:
 
 f - reweight fidelity nodes assuming that number of points per class is known
 
 w - reweight fidelity nodes assuming that number of points per class is unknown
 
 n - fidelity nodes are not reweighted
 
 
 */



int main(int argc, char *argv[]){
    int i,j;
    

    float upper_bound, lower_bound;
    


            
    srand48(time(NULL));
    char *mode=argv[1];//mode of the algorithm to run
    int k=atoi(argv[2]); //number of nearest neighbors to use
    float trainingPercentage= atof(argv[3]); //fraction of images with fixed correct labels
    int maxIters=atoi(argv[4]); //max number of iterations to run mbo
    int numTrials=atoi(argv[5]);// number of experiments to run
    float stoppingCriterion=atof(argv[6]); //algorithm ends if energy change is below this parameter
    float volumeEpsilon=atof(argv[7]);  // epsilon parameter for the auction algorithm
    int lcount=10; //number of distinct classes
    
    
    int testCount=10000; //10,000 objects in the test set
    int trainCount=60000; //60,000 objects in the training set
    int pcount=testCount+trainCount; //we just combine the two sets together
    int maxNeighbors=15;  // # of nearest neighbors to use

    
    
    indexedFloat *nearestNeighborImageData=getFileData("vl_neighbor_data_mnist"); //precomputed nearest neighbor data containing the nearest neighbors of each image and the distance between them
        
       // for(i=0; i<30; i++)
          // printf(" %f \n", nearestNeighborImageData[i].dist);
    
        
    unsigned char *dataTstLbl=getFileData("test_set_labels_mnist"); //read out the labels from the test file
    unsigned char *dataTrLbl=getFileData("training_set_labels_mnist");  //read out the labels from the training file
    
    int confusionMatrix[lcount*lcount];     // confusion matrix
    memset(confusionMatrix,0,lcount*lcount*sizeof(int));
    
    
    // reading in the ground truth labels for all the points (we have all the correct labels)
    int incorrect=0;
    float time=0;
    unsigned char *correctLabels=calloc(70000,1);
    for(i=0;i<10000;i++){
        correctLabels[i]=dataTstLbl[i+8];
    }
    for(i=0;i<60000;i++){
        correctLabels[i+10000]=dataTrLbl[i+8];
    }
        
    

    float etot=0;
    
    
    
    // running EXPERIMENTS - each experiment considers a different labeled set of the same size
    
    for(j=0;j<numTrials;j++){
        
      
        // preparing to choose the labeled data
        
        float eThisRound=0;
        int wrongThisRound=0;
        int *labels=calloc(pcount,sizeof(int)); //array that will hold all of the labels
        unsigned char *fixedLabels=calloc(pcount,1); //array that tracks which points have a fixed correct label
        int *numbers=calloc(pcount,sizeof(int)); //used for shuffling
        int counter[lcount];
        memset(counter,0,lcount*sizeof(int));
        float realRatio[lcount];
        float sampleRatio[lcount];
         memset(realRatio,0,lcount*sizeof(float));
         memset(sampleRatio,0,lcount*sizeof(float));
        for(i=0;i<pcount;i++){
            numbers[i]=i; //prepare numbers to be shuffled
        }
        int seen[lcount];
        memset(seen,0,lcount*sizeof(int));
        shuffle(numbers,pcount); //shuffle numbers
        for(i=0;i<pcount;i++){
            realRatio[correctLabels[i]]++;
            if(i/(pcount*1.0)<trainingPercentage){
                fixedLabels[numbers[i]]=1;      // these points will be considered as labeled by the algorithm
                                                  // if fixedLabels[i]=1, this point is a labeled point, otherwise fixedLabels[i]=0
                sampleRatio[correctLabels[numbers[i]]]++;
            }
             //randomly choose which images will have fixed labels
        }
       
        for(i=0;i<pcount;i++){
            
            //if a label is fixed then we assign it the correct value from the label data.....
            
            if(fixedLabels[i]){
                
                labels[i]=correctLabels[i];
                
                
            }else{
                
                if(mode[1]=='p'){//initialization methods.
                    labels[i]=-1;//initialize by percolation
                }else{
                    labels[i]=rand()%lcount;//random initialization.
                }
            }
            
            
        }
        
       
        // creating the symmetric adjacency matrix out of the nearest neighbors data
        generalConvolutionStructure g=create_symmetric_adjacency_matrix(nearestNeighborImageData, pcount, maxNeighbors, k);
        
        
        
      
        // this version of the algorithm does not use surface tensions
        float *surfaceTensions=calloc(lcount*lcount,sizeof(float));

        
        
        mbo_struct mbos; //object used by mbo.c to do convolutions
        memset(&mbos,0,sizeof(mbo_struct)); //clear memory
        mbos.nncounts=g.counts; //increasing integer array of length pcount.  nncounts[i+1]-nncounts[i] gives the number of nearest neighbors of node i.
        mbos.indicies=g.neighbors; // entries nncounts[i] through nncounts[i+1]-1 hold the indicies of the nearest neighbors of node i
        mbos.weights=g.connectionStrengths; // entries nncounts[i] through nncounts[i+1]-1 hold the weights of the nearest neighbors of node i
        
        mbos.updateList=calloc(pcount,sizeof(nodeChanged)); //allocate memory for internal workspace
        mbos.surfaceTensions=NULL;//surfaceTensions; // create a lcount*lcount symmetric matrix if surface tensions are used.
        mbos.fixedLabels=fixedLabels; //binary array of length pcount recording the nodes whose label is known
        mbos.labels=labels; //integer array of length pcount holding label of each node
        mbos.pcount=pcount; //number of nodes
        mbos.lcount=lcount; //number of labels
        mbos.stoppingCriterion=stoppingCriterion; //algorithm stops if fewer than pcount*stoppingCriterion nodes change
        mbos.maxIters=maxIters; //max number of iterations
        mbos.singleGrowth=0;//use single growth or not
        //run mbo dynamics. see mbo_convolution.c
        mbos.epsilon=volumeEpsilon;  // the epsilon parameter of the auction algorithm
        mbos.Labels= correctLabels;  // the ground truth labels for the data
        
        
        // counting the number of elements in each class (since we have the ground truth)
        int classCounts[lcount];
        memset(classCounts,0,lcount*sizeof(int));
        for(i=0;i<pcount;i++){
            classCounts[correctLabels[i]]++;
        }
        for(i=0;i<lcount;i++){
            realRatio[i]/=pcount;
            sampleRatio[i]/=(pcount*trainingPercentage);
        }
        mbos.classCounts=classCounts;
        
        
        
        mbos.k=k;
         int ix,jx;
        clock_t b,e;
        b=clock();
        int lin=0;
        
        
        // initializing the class for each element
        if(mode[1]=='v'){
            
            bellman_ford_voronoi_initialization(&mbos, g, maxNeighbors,2,lin);
        }else if(mode[1]=='c'){
            for(ix=0;ix<pcount;ix++){
                labels[ix]=correctLabels[ix];
            }
        }
       
        for(ix=0;ix<pcount;ix++){
            for(jx=g.counts[ix];jx<g.counts[ix+1];jx++){
                g.connectionStrengths[jx]=kernel(g.connectionStrengths[jx]);
            }
        }
        

         normalize_matrix(g,pcount); //make the matrix have row sum 1
        
        
         mbos.temperature= 0.02;   // parameter in a technique to help the algorithm reach global minima. If set to 0, this technique is avoided.
                                   // should be set to <=0.3 and >=0

        
        
         mbos.upper_bound= 1.0;    //  we multiply this number by the class size to get the upper bound for the size of a class
                                   // if set to 1, this indicates exact class size constraints
                                   // set to > 10 if no class size constraints desired
        
         mbos.lower_bound= 1.0;    //  we multiply this number by the class size to get the lower bound for the size of a class
                                   // if set to 1, this indicates exact class size constraints
                                    // set to > 10 if no class size constraints desired
        
        
         upper_bound= mbos.upper_bound;
         lower_bound= mbos.lower_bound;

        
        
        // running the main algorithm. Two versions: the first one incorporates temperature (to help algorithm reach global minimum)
        // and the second one does not (by setting it to 0). Temperature incorporation results in a more accurate but slower algorithm.
        
        // MAIN ALGORITHM ///////////////////////////////////////////////////////////////////////////////////
        
        if (mbos.temperature > 0.0000001){
            eThisRound=run_mbo_with_temperature(mbos, mode[0]);}
        else{
            eThisRound=run_mbo(mbos, mode[0]);}
        
         //////////////////////////////////////////////////////////////////////////////////////////////////////
        
        
        
        e=clock();
        etot+=eThisRound;
        int wrongLabels[lcount];
        memset(wrongLabels,0,lcount*sizeof(int));
        
        

        // Evaluation of the algorithm: compare results to known data
        for(i=0;i<pcount;i++){
            if(correctLabels[i]!=labels[i]){
                wrongLabels[correctLabels[i]]++;
                incorrect++;
                wrongThisRound++;
            }
            confusionMatrix[correctLabels[i]*lcount+labels[i]]++;
            
        }
        

        // computing the accuracy for this experiment
        printf("%s %i %s  %f \n", "The accuracy for round ", j+1, " is ", 100-wrongThisRound*100/(pcount*1.0));
        
        
        time+=(e-b)/(CLOCKS_PER_SEC*1.0);
        free(labels);
        free(fixedLabels);
        free(numbers);
    
    }

    
       // computing the average accuracy over all experiments (each with a different labeled set)
        printf("%s %f %s %f %f \n", "Average accuracy for ", 100*trainingPercentage, " percent of points being labeled is ", 100-100*(incorrect)/(pcount*numTrials*1.0), time/numTrials);

    free(dataTrLbl);
    free(dataTstLbl);
    

    
}



