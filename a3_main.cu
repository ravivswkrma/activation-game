/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */
#include<iostream>
#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"

using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/


//Traversing the level zero and increamenting the indegree of nodes connected to them
__global__ void traverseLevelZero(int *csr,int *offset, bool *isActive, int *aid, int *level, int vertices, int cnst, int currLvl)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id<vertices && level[id]==currLvl)
    {
        
        isActive[id]=true;

        aid[id]=0;

        int sidx = offset[id];
        int eidx = offset[id+1];

        for(int i=sidx; i<eidx; i++)
            atomicCAS(&level[csr[i]],-1,currLvl+1);
        

        for(int i=sidx; i<eidx; i++)
            atomicInc((unsigned int *)&aid[csr[i]], 15000); 
        
    }

}


__global__ void levelZero(int *apr, int *level, int v,int cnst)
{
    int id = blockIdx.x *blockDim.x + threadIdx.x;

    if(id<v && apr[id]==0)
        level[id]=0;    
}

//This kernel applies the activation rule i.e the first rule on nodes
__global__ void activatingVertices( int *offset, int *csr, int *level, int *aid, int *apr, bool *isActive,int V,int currLvl, int cnst, int edges)
{ 
    
    int id = blockIdx.x* blockDim.x + threadIdx.x;

    int sidx = offset[id];

    int eidx = offset[id+1];

    if(id < V && level[id] == currLvl)
    {
        for(int i=sidx; i<eidx; i++)
            atomicCAS(&level[csr[i]],-1,currLvl+1); 

        if(apr[id] <= aid[id])
            isActive[id]=true;   
        
    }
}
 //This kernel applies the deactivation rule i.e the first rule on nodesd   
__global__ void deactivatingVertices(int *offset, int *csr, int *level, int *aid, int *apr, bool *isActive,int V,int currLvl, int cnst, int edges)
{ 
    
    int id = blockIdx.x *blockDim.x + threadIdx.x;

    if(id<V && level[id]==currLvl)
    {
        //Apply Deactivation rule
        if((id-1)>=0 && (id+1)<V && level[id-1]==currLvl && level[id+1]==currLvl && isActive[id+1] !=true && isActive[id-1] != true )
            isActive[id]=false;
        
        //Checking whether the node is already active if active then increament the indegree of next level nodes which are connected to it
        if(isActive[id])
            for(int i=offset[id]; i<offset[id+1];i++)
                atomicInc((unsigned int*)&aid[csr[i]] ,  15000);
            
    }
}

//Applying activation and deactivation on the last level
__global__ void lastLevel(int *level, int *aid, int *apr, bool *isActive,int V,int currLvl, int cnst)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if(id<V && level[id]==currLvl)
    {
        // Apply activation rule
        if(aid[id]>= apr[id]){
            isActive[id]=true;   
      }

        //Apply Deactivation rule
        if((id-1)>=0 && (id+1)<V && level[id-1]==currLvl && level[id+1]==currLvl && isActive[id-1]!=true && isActive[id+1]!=true)
            isActive[id]=false;    
    }
}
    
//Finally calculating the result
__global__ void solve(bool *isActive, int *level, int *result,int vetex, int cnst) 
{  
    int id = blockIdx.x * blockDim.x + threadIdx.x;

        if(id<vetex && isActive[id])
            atomicInc((unsigned int *) &result[level[id]] , 15000);
}

/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    int *d_activeVertex;
    cudaMalloc(&d_activeVertex, L*sizeof(int));
    cudaMemset(d_activeVertex, 0, L*sizeof(int));

/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/

// Array for storing AID of each vertex.
cudaMemset(d_aid, 0, V*sizeof(int));


//This array keeps track of whether the vetex is active or not
bool *d_isActive;
cudaMalloc(&d_isActive, V*sizeof(bool));
cudaMemset(d_isActive, 0, V*sizeof(bool));

//It stores the level of vetices
int *d_levelOfVertex; // We will inialize it to -1
cudaMalloc(&d_levelOfVertex, V*sizeof(int));
cudaMemset(d_levelOfVertex, -1, V*sizeof(int));

//No. of threads each block can have 
int blockSize = 512;

//Kernel Configuration starts
int numBlocks = (V+blockSize)/blockSize;
levelZero<<<numBlocks,blockSize>>>(d_apr,d_levelOfVertex, V, blockSize);
cudaDeviceSynchronize();


//Now we know the level 0 nodes 


/*This kernel processes level 0*/
traverseLevelZero<<<numBlocks,blockSize>>>(d_csrList, d_offset, d_isActive, d_aid, d_levelOfVertex, V, blockSize, 0); 
cudaDeviceSynchronize();

/*This kernel process level 1 to l-1 */

for(int i=1; i<L-1; i++)
{
    //It activates the vetices i.e applies rule 1
    activatingVertices<<<numBlocks,blockSize>>>(d_offset, d_csrList,d_levelOfVertex, d_aid, d_apr, d_isActive,V,i,blockSize, E); 
    cudaDeviceSynchronize();


   //It deactivates the vetices i.e applies rule 2
    deactivatingVertices<<<numBlocks,blockSize>>>(d_offset, d_csrList,d_levelOfVertex, d_aid, d_apr, d_isActive,V,i,blockSize,E);
    cudaDeviceSynchronize();
}

/* Last level is processed here */

lastLevel<<<numBlocks,blockSize>>>(d_levelOfVertex, d_aid, d_apr, d_isActive,V, L-1,blockSize); // send last level parameter
cudaDeviceSynchronize();
       
/*This kernel calculates the final answer */
solve<<<numBlocks,blockSize>>>(d_isActive, d_levelOfVertex, d_activeVertex,V,blockSize); //level, active     
cudaDeviceSynchronize();

/*device to host transfer*/
cudaMemcpy(h_activeVertex, d_activeVertex, L*sizeof(int), cudaMemcpyDeviceToHost);


/********************************END OF CODE AREA**********************************/


double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
