#include "pairHMM.cuh"

#define NOW std::chrono::high_resolution_clock::now();

#define CHECK(ans) { GPUAssert((ans), __FILE__, __LINE__); }
#define CHECK_KERNELCALL()                                                            \
{                                                                                     \
    const cudaError_t err = cudaGetLastError();                                       \
    if (err != cudaSuccess)                                                           \
    {                                                                                 \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                           \
    }                                                                                 \
}

extern __constant__ __half2 ph2pr_d;

inline void GPUAssert(cudaError_t code, 
                      const char *file, 
                      int line, 
                      bool abort=true) 
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU assert: %s - file: %s, line: %d\n", cudaGetErrorString(code), file, line);
      if (abort) 
        exit(code);
   }
}

int8_t* ptrStreams_d[N_STREAMS];
int8_t* ptrStreams_h[N_STREAMS];
cudaStream_t *streams = (cudaStream_t *)malloc(N_STREAMS * sizeof(cudaStream_t));

void organize_input(std::vector<uint64_t>& memPerAlignment, 
                    std::vector<std::string>& read, 
                    std::vector<std::string>& qual_read, 
                    std::vector<std::string>& qual_del, 
                    std::vector<std::string>& qual_ins, 
                    std::vector<std::string>& hap, 
                    std::vector<int>& rlen, 
                    std::vector<int>& hlen, 
                    uint64_t memPerStream, 
                    int streamID, 
                    int start_alignment, 
                    int processed, 
                    uint64_t no_couples,
                    int max_len_overall, 
                    std::vector<uint64_t>& alignmentsPerStream, 
                    std::vector<uint64_t>& matrixSize_gm, 
                    std::vector<uint64_t>& matrixSize_sm, 
                    std::vector<uint64_t>& tot_hlen, 
                    std::vector<uint64_t>& tot_rlen, 
                    std::vector<uint64_t>& offset_results
                    /*int8_t** ptrStreams_h*/)
{
  alignmentsPerStream[streamID] = 0;           // Represents the number of blocks for the GPU
  matrixSize_gm[streamID] = 0;
  matrixSize_sm[streamID] = 0;
  tot_hlen[streamID] = 0;
  tot_rlen[streamID] = 0;

  uint64_t count_alignments = 0;
  uint64_t requiredMem = 0;
  uint64_t memPair = 0;
  uint64_t maxRead;
  uint64_t maxHap;

  memPair = memPerAlignment[start_alignment + count_alignments] > memPerAlignment[start_alignment + count_alignments + 1]  ? memPerAlignment[start_alignment + count_alignments]*2 : memPerAlignment[start_alignment + count_alignments + 1]*2;
  
  while (((requiredMem + memPair) < memPerStream) && ((processed + count_alignments + 2) <= no_couples)) {
    requiredMem += memPair; 
    maxHap = hlen[start_alignment + count_alignments] > hlen[start_alignment + count_alignments + 1] ? hlen[start_alignment + count_alignments] : hlen[start_alignment + count_alignments + 1];
    maxRead = rlen[start_alignment + count_alignments] > rlen[start_alignment + count_alignments + 1] ? rlen[start_alignment + count_alignments] : rlen[start_alignment + count_alignments + 1];
    tot_hlen[streamID] += ((maxHap + 1) * 2); // (hlen[start_alignment + count_alignments] + hlen[start_alignment + count_alignments + 1]);
    tot_rlen[streamID] += ((maxRead + 1) * 2); // (rlen[start_alignment + count_alignments] + rlen[start_alignment + count_alignments + 1]);
    count_alignments += 2;
    memPair = memPerAlignment[start_alignment + count_alignments] > memPerAlignment[start_alignment + count_alignments + 1]  ? memPerAlignment[start_alignment + count_alignments]*2 : memPerAlignment[start_alignment + count_alignments + 1]*2;
  } 

  // std::cerr << "Stream " << streamID << " - scheduled alignments: " << count_alignments << ", requiredMem: " << requiredMem << ", memPerStream:" << memPerStream << std::endl;

  tot_rlen[streamID] = ceil((tot_rlen[streamID] + 1)/4.0) * 4;
  tot_hlen[streamID] = ceil((tot_hlen[streamID] + 1)/4.0) * 4;

  alignmentsPerStream[streamID] = count_alignments;

  assert((sizeof(char) * tot_rlen[streamID] +
          sizeof(char) * tot_rlen[streamID] * 3 + 
          sizeof(char) * tot_hlen[streamID] +
          sizeof(int) * alignmentsPerStream[streamID] * 5 + 
          sizeof(restype) * alignmentsPerStream[streamID]) < memPerStream);

  offset_results[streamID] =  sizeof(char) * tot_rlen[streamID] +
                              sizeof(char) * tot_rlen[streamID] * 3 + 
                              sizeof(char) * tot_hlen[streamID] +
                              sizeof(int) * alignmentsPerStream[streamID] * 5;

  // Pointers to store inputs in pinned memory 
  char* read_h = (char *)ptrStreams_h[streamID];                                                //[batchID % N_STREAMS];
  char* q_reads_h = (char *)(read_h + tot_rlen[streamID]); 
  char* q_ins_h = q_reads_h + tot_rlen[streamID];
  char* q_del_h = q_ins_h + tot_rlen[streamID];
  char* hap_h = (char *)(q_del_h + tot_rlen[streamID]);
  int* rlen_h  = (int *)(hap_h + tot_hlen[streamID]);
  int* hlen_h = rlen_h + alignmentsPerStream[streamID];
  int* overLimit = hlen_h + alignmentsPerStream[streamID];                                      // = (int *)malloc(sizeof(int) * no_couples);
  int* prefixsum_read_h = overLimit + alignmentsPerStream[streamID];                            // = (int *)malloc(sizeof(int) * (no_reads));
  int* prefixsum_hap_h = prefixsum_read_h + alignmentsPerStream[streamID];                      // = (int *)malloc(sizeof(int) * (no_haps));
  restype* results_float_h = (restype *)(prefixsum_hap_h + alignmentsPerStream[streamID]);      // = (float *)malloc(sizeof(float) * (no_couples));
  matricestype* matrixes_float_h = (matricestype*)(results_float_h + alignmentsPerStream[streamID]); 

  uint64_t idx_read = 0;
  uint64_t idx_hap = 0;
  uint64_t idx_len = 0;

  CHECK(cudaMemset(ptrStreams_h[streamID], 'I' - Q_OFFSET, sizeof(int8_t) * memPerStream)); 

  // std::cerr << "------------------------------------------------------------------" << std::endl;

  // Copy data to pinned memory
  for (int j = start_alignment; j < start_alignment + alignmentsPerStream[streamID]; j += 2)
  {

    maxRead = rlen[j] > rlen[j + 1] ? rlen[j] : rlen[j + 1];
    maxHap = hlen[j] > hlen[j + 1] ? hlen[j] : hlen[j + 1];

    prefixsum_read_h[j] = idx_read;
    prefixsum_read_h[j + 1] = idx_read;
    prefixsum_hap_h[j] = idx_hap;
    prefixsum_hap_h[j + 1] = idx_hap;

    rlen_h[idx_len] = rlen[j];
    hlen_h[idx_len] = hlen[j];
    idx_len++;

    rlen_h[idx_len] = rlen[j + 1];
    hlen_h[idx_len] = hlen[j + 1];
    idx_len++;

    int idx_x = 0;
    int idx_y = 1;

    // Reads stored in order
    // Aligned left
    // Read
    for (int l = 0; l < rlen[j]; l++){
      read_h[idx_read + idx_x] = read[j][l]; // read[j][rlen[j] - 1 - l];
      idx_x += 2;
    }

    for (int l = 0; l < rlen[j + 1]; l++){
      read_h[idx_read + idx_y] = read[j + 1][l]; // read[j][rlen[j] - 1 - l];
      idx_y += 2;
    }

    // Qualities
    idx_x = 0;
    idx_y = 1;
    for (int l = 0; l < rlen[j]; l++){
      q_reads_h[idx_read + idx_x] = qual_read[j][l] - Q_OFFSET; // qual_read[j][rlen[j] - 1 - l] - Q_OFFSET; // powf(10.0f, (- (qual_read[j][rlen[j] - 1 - l] - Q_OFFSET) * 0.1f));
      idx_x += 2;
    }

    for (int l = 0; l < rlen[j + 1]; l++){
      q_reads_h[idx_read + idx_y] = qual_read[j + 1][l] - Q_OFFSET; // qual_read[j][rlen[j] - 1 - l] - Q_OFFSET; // powf(10.0f, (- (qual_read[j][rlen[j] - 1 - l] - Q_OFFSET) * 0.1f));
      idx_y += 2;
    }

    // Insertion
    idx_x = 0;
    idx_y = 1;

    for (int l = 0; l < rlen[j]; l++){
      q_ins_h[idx_read + idx_x] = qual_ins[j][l] - Q_OFFSET; // qual_ins[j][rlen[j] - 1 - l] - Q_OFFSET; // powf(10.0f, (- (qual_ins[j][rlen[j] - 1 - l] - Q_OFFSET) * 0.1f));
      idx_x += 2;
    }

    for (int l = 0; l < rlen[j + 1]; l++){
      q_ins_h[idx_read + idx_y] = qual_ins[j + 1][l] - Q_OFFSET; // qual_ins[j][rlen[j] - 1 - l] - Q_OFFSET; // powf(10.0f, (- (qual_ins[j][rlen[j] - 1 - l] - Q_OFFSET) * 0.1f));
      idx_y += 2;
    }

    // Deletion
    idx_x = 0;
    idx_y = 1;

    for (int l = 0; l < rlen[j]; l++){
      q_del_h[idx_read + idx_x] = qual_del[j][l] - Q_OFFSET; // qual_del[j][rlen[j] - 1 - l] - Q_OFFSET; // powf(10.0f, (- (qual_del[j][rlen[j] - 1 - l] - Q_OFFSET) * 0.1f));
      idx_x += 2;
    }

    for (int l = 0; l < rlen[j + 1]; l++){
      q_del_h[idx_read + idx_y] = qual_del[j + 1][l] - Q_OFFSET; // qual_del[j][rlen[j] - 1 - l] - Q_OFFSET; // powf(10.0f, (- (qual_del[j][rlen[j] - 1 - l] - Q_OFFSET) * 0.1f));
      idx_y += 2;
    }

    idx_read += (maxRead * 2); // (rlen[j] + rlen[j + 1]); 

    // Haplotype
    idx_x = 0;
    idx_y = 1;

    int shift = (maxHap - hlen[j]) * 2;

    for (int l = 0; l < hlen[j]; l++){
      hap_h[idx_hap + shift + idx_x] = hap[j][hlen[j] - 1 - l]; // hap[j][l];
      idx_x += 2;
    }

    shift = (maxHap - hlen[j + 1]) * 2;

    for (int l = 0; l < hlen[j + 1]; l++){
      hap_h[idx_hap + shift + idx_y] = hap[j + 1][hlen[j + 1] - 1 - l]; // hap[j][l];
      idx_y += 2;
    }

    idx_hap += (maxHap * 2); // (hlen[j] + hlen[j + 1]);

  }

  // for(int p = 24; p < 74; p++){
  //   std::cout << read_h[p];
  // }
  // std::cout << std::endl;

  // ------------------------------------------------------------------
  // Assessing which alignment needs global memory
  int counter_overlimit = 0;
  int maxlen_gm = 0;
  int maxlen_sm = 0;

  for (int j = 0; j < alignmentsPerStream[streamID]; j++)
  {
    int hlen = hlen_h[j];
    maxlen_gm = (hlen > (MAX_SUPPORTED_LENGTH) && hlen > maxlen_gm) ? hlen : maxlen_gm;
    maxlen_sm = (hlen <= (MAX_SUPPORTED_LENGTH)  && hlen >= maxlen_sm) ? hlen : maxlen_sm;
    overLimit[j] = hlen > (MAX_SUPPORTED_LENGTH) ? counter_overlimit++ : (-1);
  }

  matrixSize_sm[streamID] = maxlen_sm; // (CACHE_SIZE * (maxlen_sm + 1)); // one for the added column
  matrixSize_gm[streamID] = maxlen_gm; // (CACHE_SIZE * (maxlen_gm + 1)); // one for the added column

  // std::cerr << "\t\t" << tot_hlen[streamID] << "\t\t" << tot_rlen[streamID] << std::endl;
  // std::cerr << "\t\t" << matrixSize_sm[streamID] << "\t\t" << matrixSize_gm[streamID] << std::endl;

}

void launch_stream(dim3 numThreads, 
                   uint64_t memPerStream, 
                   int streamID, 
                   std::vector<uint64_t>& alignmentsPerStream, 
                   std::vector<uint64_t>& matrixSize_gm, 
                   std::vector<uint64_t>& matrixSize_sm, 
                   std::vector<uint64_t>& tot_hlen, 
                   std::vector<uint64_t>& tot_rlen, 
                   std::vector<uint64_t>& offset_results
                   /*int8_t** ptrStreams_h, 
                   int8_t** ptrStreams_d, 
                   cudaStream_t* streams*/)
{

  uint64_t shmem_size = 3 * CACHE_SIZE * (matrixSize_sm[streamID] + 1) * sizeof(matricestype) +  // matrices in __half2
                        10 * matrixSize_sm[streamID];                 // 2x sequences (2 reads, 2 haps, 2 qread, 2 qins, 2 qdel)
  
  // Number of blocks is the half the number of scheduled alignments, rounded up
  dim3 numBlocks(alignmentsPerStream[streamID] / 2, 1, 1); 

  CHECK(cudaMemcpyAsync(ptrStreams_d[streamID], ptrStreams_h[streamID], sizeof(int8_t) * offset_results[streamID], cudaMemcpyHostToDevice, streams[streamID]));

  // std::cerr << "\t\tStream params: " << F16_MAX << " " << matrixSize_sm[streamID] << " " << matrixSize_gm[streamID] << " " << tot_hlen[streamID] << " " << tot_rlen[streamID] << " " << alignmentsPerStream[streamID] << std::endl;

  pairHMM<<<numBlocks, numThreads, shmem_size, streams[streamID]>>>(F16_MAX,
                                                                    matrixSize_sm[streamID],
                                                                    matrixSize_gm[streamID],
                                                                    ptrStreams_d[streamID],    // [batchID % N_STREAMS],
                                                                    tot_hlen[streamID],
                                                                    tot_rlen[streamID],
                                                                    alignmentsPerStream[streamID]);

  
  CHECK_KERNELCALL();

  // CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpyAsync(ptrStreams_h[streamID] + offset_results[streamID], ptrStreams_d[streamID] + offset_results[streamID], sizeof(restype) * alignmentsPerStream[streamID], cudaMemcpyDeviceToHost, streams[streamID]));

}

void sync_and_retrieve_results(int streamID, 
                               std::vector<restype>& results_final, 
                               std::vector<uint64_t>& alignmentsPerStream, 
                               std::vector<uint64_t>& matrixSize_gm, 
                               std::vector<uint64_t>& matrixSize_sm, 
                               std::vector<uint64_t>& tot_hlen, 
                               std::vector<uint64_t>& tot_rlen, 
                               std::vector<uint64_t>& offset_results
                               /*int8_t** ptrStreams_h, 
                               int8_t** ptrStreams_d, 
                               cudaStream_t* streams*/)
{

  restype* res_local = (restype*)(ptrStreams_h[streamID % N_STREAMS] + offset_results[streamID % N_STREAMS]);

  // Data transfer from GPU
  CHECK(cudaStreamSynchronize(streams[streamID % N_STREAMS]));
  
  CHECK(cudaPeekAtLastError());
  
  for (int j = 0; j < alignmentsPerStream[streamID % N_STREAMS]; j++)
  {
    results_final.push_back(log10f(static_cast<double>(res_local[j])) - log10f(static_cast<double>(F16_MAX)));// log10f(static_cast<double>(INIT))); //__float2half(INIT))));
  }

  CHECK(cudaPeekAtLastError());

}

int main (int argc, char* argv[]) 
{

  if (argc < 3)
  {
    std::cerr << "./demo path/to/input deviceID"<<std::endl;
    exit(1);
  }

  std::string input_path = argv[1];
  dim3 numThreads(64, 1, 1); // (atoi(argv[2]), 1, 1);
  unsigned int deviceID = (atoi(argv[2])); // atoi(argv[3]);
  
  std::ifstream f_input(input_path.c_str());

  CHECK(cudaSetDevice(deviceID));

  CHECK(cudaFree(0));
  
  // cudaDeviceProp prop;
  uint64_t total_mem, free_mem;
  CHECK(cudaMemGetInfo(&free_mem, &total_mem));

  uint64_t memPerStream = floor(((free_mem*90)/100)/N_STREAMS);

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, deviceID));
  std::cerr << "Shared memory per block: " << prop.sharedMemPerBlock << " B" << std::endl;
  std::cerr << "Max allocated shared memory per block: " << (MAX_SUPPORTED_LENGTH * sizeof(matricestype) * 3 * 3) + (MAX_SUPPORTED_LENGTH * 10) << " B" << std::endl;

  std::vector<uint64_t> alignmentsPerStream(N_STREAMS);
  std::vector<uint64_t> matrixSize_gm(N_STREAMS);
  std::vector<uint64_t> matrixSize_sm(N_STREAMS);
  std::vector<uint64_t> tot_hlen(N_STREAMS);
  std::vector<uint64_t> tot_rlen(N_STREAMS);
  std::vector<uint64_t> offset_results(N_STREAMS);

  // int8_t* ptrStreams_d[N_STREAMS];
  // int8_t* ptrStreams_h[N_STREAMS];
  // cudaStream_t* streams = (cudaStream_t *)malloc(N_STREAMS * sizeof(cudaStream_t));
  
  for (int i = 0; i < N_STREAMS; i++)
  {
    CHECK(cudaStreamCreate(&streams[i]));
    CHECK(cudaMalloc(&ptrStreams_d[i], sizeof(int8_t) * memPerStream));
    CHECK(cudaMallocHost(&ptrStreams_h[i], sizeof(int8_t) * memPerStream));
    CHECK(cudaMemset(ptrStreams_d[i], 'I' - Q_OFFSET, sizeof(int8_t) * memPerStream)); 
    CHECK(cudaMemset(ptrStreams_h[i], 'I' - Q_OFFSET, sizeof(int8_t) * memPerStream)); 
  }

  // memPerStream = floor(memPerStream * 0.8); // to be sure to have more memory

  // Support vectors to store inputs from input file
  std::vector<std::string> read, hap;
  std::vector<std::string> qual_read, qual_ins, qual_del, qual_gcp;
  std::vector<int> rlen;                         // Read lengths
  std::vector<int> hlen;                         // Haplotype lengths
  std::vector<restype> results_final;        
  
  std::string tmp;

  uint64_t no_couples = 0;

  int no_reads_tmp, no_haps_tmp;
  int processed = 0;
  int start_alignment = 0;
  int max_len_overall = 0;
  int streamID = 0;

  //--------------------------------------------------
  //  Reading input file and duplicating alignments
  //

  std::cerr << "Reading file...\n";

  while(!f_input.eof())
  {
    
    // Reading number of reads and haplotypes
    std::getline(f_input, tmp);

    std::stringstream sstream(tmp);
    sstream >> no_reads_tmp;
    sstream >> no_haps_tmp;
    no_couples += (no_reads_tmp * no_haps_tmp);
    
    // Reading and duplicating all the reads and their quality scores
    for(int i = 0; i < no_reads_tmp; i++) 
    {
      std::string read_tmp, hap, qread, qins, qdel;
      std::getline(f_input, tmp);
      std::stringstream sstream(tmp);
      sstream >> read_tmp;
      sstream >> qread;
      sstream >> qins;
      sstream >> qdel;
      for (int j = 0; j < no_haps_tmp; j++)
      {
        read.push_back(read_tmp);
        rlen.push_back(read_tmp.size());
        qual_read.push_back(qread);
        qual_ins.push_back(qins);
        qual_del.push_back(qdel); 
      }
      // sstream >> tmp_read;
    }

    std::vector<std::string> hap_tmp;

    // Read and duplicate all the haplotypes
    for(int i = 0; i < no_haps_tmp; i++) 
    {
      std::getline(f_input, tmp);
      std::stringstream sstream(tmp);
      hap_tmp.push_back(tmp);
      tmp.clear();
    }

    for (int i = 0; i < no_reads_tmp; i++)
    {
      for(int j = 0; j < no_haps_tmp; j++)
      {
        tmp.clear();
        std::string tmp = hap_tmp[j];
        hap.push_back(tmp);
        int len = tmp.size();
        if (len > max_len_overall) max_len_overall = len;
        hlen.push_back(len);
      }
    }


  }

  std::cerr << "Done reading file!\n";

  max_len_overall = ceil((max_len_overall + 1)/4.0) * 4 + 4;

  int real_no_couples = no_couples;

  // Duplicating last alignment when total alignments are odd
  if (no_couples % 2 != 0){
    read.push_back(read[no_couples - 1]);
    hap.push_back(hap[no_couples - 1]);
    rlen.push_back(rlen[no_couples - 1]);
    hlen.push_back(hlen[no_couples - 1]);
    qual_read.push_back(qual_read[no_couples - 1]);
    qual_ins.push_back(qual_ins[no_couples - 1]);
    qual_del.push_back(qual_del[no_couples - 1]);
    no_couples++;
  }

  // std::cerr << "max_len_overall: " << max_len_overall << std::endl;

  //--------------------------------------------------
  //  Computing required memory for each alignment
  //
  std::vector<uint64_t> memPerAlignment(no_couples);
  for (int i = 0; i < no_couples; i++){
    uint64_t maxlen = rlen[i] > hlen[i] ? rlen[i] : hlen[i];
    uint64_t mem = (ceil(sizeof(char) * maxlen / 4.0) * 4) +                                  // Read
                   (ceil(sizeof(char) * maxlen / 4.0) * 4) +                                  // Haplotype
                   (ceil(sizeof(char) * maxlen / 4.0) * 4) * 3 +                              // Qualities
                   sizeof(restype) +                                                          // Result
                   sizeof(int) * 4 +                                                          // Lengths & prefix sums
                   sizeof(int) +                                                              // Counter for global memory
                   sizeof(matricestype) * ((ceil(sizeof(char) * maxlen / 4.0) * 4) > MAX_SUPPORTED_LENGTH) * 3 * CACHE_SIZE * (max_len_overall);
    memPerAlignment[i] = mem;
  }

  //--------------------------------------------------
  //  Fill constant memory with pre-computed scores
  //
  __half2 ph2pr_full_h[PH2PR_SIZE * PH2PR_SIZE]; 
  
  // Fill array with all possible combinations
  for(int i = 0; i < PH2PR_SIZE; i++){
    __half n1 = __float2half(powf(10.0f, (- i * 0.1f)));
    for(int j = 0; j < PH2PR_SIZE; j++){
      __half n2 = __float2half(powf(10.0f, (- j * 0.1f)));
      ph2pr_full_h[i * PH2PR_SIZE + j].x = n1;
      ph2pr_full_h[i * PH2PR_SIZE + j].y = n2;
    }
  }

  // Copy data to constant memory
  CHECK(cudaMemcpyToSymbol(ph2pr_d, ph2pr_full_h, sizeof(matricestype) * PH2PR_SIZE * PH2PR_SIZE));

  //--------------------------------------------------
  //  Analyze each batch on a different stream
  //
  for (int launch = 0; launch < 30; launch++){
  auto start = NOW;
  processed = 0;
  start_alignment = 0;
  streamID = 0;

  // Fill each available stream till there are alignments 
  for (int i = 0; i < N_STREAMS && processed < no_couples; i++)
  {
    // std::cerr << "First loop - Stream: " << streamID << std::endl;

    organize_input(memPerAlignment, read, qual_read, qual_del, qual_ins, hap, rlen,  hlen, memPerStream, i, start_alignment, processed, no_couples, max_len_overall, alignmentsPerStream, matrixSize_gm, matrixSize_sm, tot_hlen, tot_rlen, offset_results /*ptrStreams_h*/);
    
    launch_stream(numThreads, memPerStream, i, alignmentsPerStream, matrixSize_gm, matrixSize_sm, tot_hlen, tot_rlen, offset_results /*, ptrStreams_h, ptrStreams_d, streams*/);

    start_alignment += alignmentsPerStream[i];
    processed += alignmentsPerStream[i];

    // std::cerr << "batch: " << streamID << "\tprocessed: " << processed << "/" << no_couples << std::endl;

    streamID++;

  }

  // Continue processing till batches are available
  while(processed < no_couples)
  {

    // std::cerr << "Middle loop - Stream: " << streamID << std::endl;

    sync_and_retrieve_results(streamID % N_STREAMS, results_final, alignmentsPerStream, matrixSize_gm, matrixSize_sm, tot_hlen, tot_rlen, offset_results /*, ptrStreams_h, ptrStreams_d, streams*/);

    organize_input(memPerAlignment, read, qual_read, qual_del, qual_ins, hap, rlen,  hlen, memPerStream, streamID % N_STREAMS, start_alignment, processed, no_couples, max_len_overall, alignmentsPerStream, matrixSize_gm, matrixSize_sm, tot_hlen, tot_rlen, offset_results /*, ptrStreams_h*/);

    launch_stream(numThreads, memPerStream, streamID % N_STREAMS, alignmentsPerStream, matrixSize_gm, matrixSize_sm, tot_hlen, tot_rlen, offset_results /*, ptrStreams_h, ptrStreams_d, streams*/);

    start_alignment += alignmentsPerStream[streamID % N_STREAMS];
    processed += alignmentsPerStream[streamID % N_STREAMS];

    // std::cerr << "batch: " << streamID << "\tprocessed: " << processed << "/" << no_couples << std::endl;
    
    streamID++;    

  }

  // Retrieve remaining results
  // std::cerr << "Retrieve remaining results\n";

  // Number of streams to free
  int n_iter = streamID < N_STREAMS ? streamID : N_STREAMS;
  streamID = streamID > N_STREAMS ? streamID - N_STREAMS : 0;

  for (int i = 0; i < n_iter ; i++)
  {
    // std::cerr << "Last loop - Stream: " << streamID << std::endl;
    sync_and_retrieve_results(streamID, results_final, alignmentsPerStream, matrixSize_gm, matrixSize_sm, tot_hlen, tot_rlen, offset_results /*, ptrStreams_h, ptrStreams_d, streams*/);

    streamID++;
  }

  auto end = NOW;
  std::chrono::duration<double> time = (end - start);
  std::cout << "Time: " << time.count() << std::endl;

  }
  // std::cout << time.count() << ", ";
  // // std::cerr << "Clearing memory\n";

  // double seconds = time.count();
  // // std::cout << seconds << std::endl;

  // // std::cout << rlen[0] << " " << hlen[0] << std::endl;
  // double GCUPS = (rlen[0] * hlen[0]) * no_couples / (seconds * 1000 * 1000000);
  // std::cout << GCUPS << ", \n";
  
  // Free memory and streams
  for (int i = 0; i < N_STREAMS; i++)
  {
    CHECK(cudaFreeHost(ptrStreams_h[i]));
    // CHECK(cudaFree(ptrStreams_d[i], streams[i]));
    CHECK(cudaFree(ptrStreams_d[i]));
    CHECK(cudaStreamDestroy(streams[i]));
  }

  // for (int i = 0; i < real_no_couples /*results_final.size()*/; i++)
  // {
    std::cout << std::setprecision(5) << std::fixed;
    std::cout << results_final[0] << "\n";
  // }

  return 0;
  
}