importScripts('./onnxruntime-web/ort.wasm.min.js');

ort.env.wasm.wasmPaths = '/onnxruntime-web/';
ort.env.wasm.proxy = true; 
ort.env.wasm.simd = true; 
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

let session = null;
let tokenizer = null;
let isInitialized = false;

// Add custom error class at the top
class WorkerError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = 'WorkerError';
    this.cause = options.cause || null;
    this.type = options.type || 'unknown';
    this.details = options.details || {};
    
    // Ensure these properties are always defined
    this.filename = options.filename || 'inference_worker.js';
    this.lineno = options.lineno || -1;
    this.message = message || 'An unknown error occurred';
  }

  toJSON() {
    return {
      name: this.name,
      message: this.message,
      type: this.type,
      cause: this.cause?.message,
      stack: this.stack,
      filename: this.filename,
      lineno: this.lineno,
      details: this.details
    };
  }
}

async function init() {
  try {
    console.log('Starting worker initialization...');    

    const initPromise = (async () => {
        
      // Initialize ONNX Runtime Web
      if (typeof ort.InferenceSession === 'undefined') {
        throw new WorkerError('ONNX Runtime not loaded properly', {
          type: 'initialization_error'
        });
      }
    
      try {
        session = await ort.InferenceSession.create('model/model_O4.onnx', {          
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        });
      } catch (error) {
        throw new WorkerError('Failed to create inference session', {
          type: 'session_error',
          cause: error,
          details: { sessionOptions: { executionProviders: ['wasm'] } }
        });
      }

      try {
        const tokenizerResponse = await fetch('model/tokenizer.json');
        if (!tokenizerResponse.ok) {
          throw new WorkerError('Failed to fetch tokenizer', {
            type: 'tokenizer_fetch_error',
            details: { status: tokenizerResponse.status }
          });
        }
        
        const tokenizerJSON = await tokenizerResponse.json();
        if (!tokenizerJSON?.model?.vocab) {
          throw new WorkerError('Invalid tokenizer format', {
            type: 'tokenizer_format_error',
            details: { receivedData: tokenizerJSON }
          });
        }
        
        tokenizer = new SimpleTokenizer(tokenizerJSON.model.vocab);
      } catch (error) {
        throw new WorkerError('Tokenizer initialization failed', {
          type: 'tokenizer_error',
          cause: error
        });
      }
    })();

    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => {
        reject(new WorkerError('Initialization timed out after 30 seconds', {
          type: 'timeout_error'
        }));
      }, 30000);
    });

    await Promise.race([initPromise, timeoutPromise]);
    
    isInitialized = true;
    postMessage({ type: 'init-complete' });
  } catch (error) {
    // Ensure we're working with our custom error type
    const workerError = error instanceof WorkerError 
      ? error 
      : new WorkerError(error.message || 'Unknown error', {
          cause: error,
          type: 'uncaught_error'
        });

    console.error('Worker error:', workerError);
    
    postMessage({ 
      type: 'error',
      error: workerError.message,
      details: workerError.toJSON()
    });
    
    throw workerError;
  }
}

async function getEmbedding(text) {
  try {
    // Convert text -> Tensors
    const encoding = tokenizer.encode(text);
    const input_ids = new ort.Tensor(
      'int64',
      BigInt64Array.from(encoding.ids.map(BigInt)),
      [1, encoding.ids.length]
    );
    const attention_mask = new ort.Tensor(
      'int64',
      BigInt64Array.from(encoding.attentionMask.map(BigInt)),
      [1, encoding.attentionMask.length]
    );
    const token_type_ids = new ort.Tensor(
      'int64',
      BigInt64Array.from(encoding.tokenTypeIds.map(BigInt)),
      [1, encoding.tokenTypeIds.length]
    );
    const feeds = {
      input_ids,
      attention_mask,
      token_type_ids
    };

    // Inference
    const results = await session.run(feeds);
    const outputKey = Object.keys(results)[0];
    const outputData = results[outputKey].data;
    const [batchSize, seqLength, hiddenSize] = results[outputKey].dims;

    // Parallelize embedding calculations using Web Workers
    const chunkSize = Math.ceil(seqLength / (navigator.hardwareConcurrency || 4));
    let sumEmbedding = new Float32Array(hiddenSize).fill(0);
    let validTokens = 0;
    const maskData = attention_mask.data;

    // Process chunks in parallel
    const chunks = [];
    for (let start = 0; start < seqLength; start += chunkSize) {
      const end = Math.min(start + chunkSize, seqLength);
      chunks.push({start, end});
    }

    const resultsParallel = await Promise.all(chunks.map(async ({start, end}) => {
      let localSum = new Float32Array(hiddenSize).fill(0);
      let localValidTokens = 0;

      for (let i = start; i < end; i++) {
        if (Number(maskData[i]) === 1) {
          localValidTokens++;
          for (let j = 0; j < hiddenSize; j++) {
            localSum[j] += outputData[i * hiddenSize + j];
          }
        }
      }
      return { sum: localSum, validTokens: localValidTokens };
    }));

    // Combine results
    resultsParallel.forEach(({sum, validTokens: localValidTokens}) => {
      for (let i = 0; i < hiddenSize; i++) {
        sumEmbedding[i] += sum[i];
      }
      validTokens += localValidTokens;
    });

    const meanEmbedding = Array.from(sumEmbedding).map(v => v / validTokens);

    // Normalize using optimized array operations
    const norm = Math.sqrt(meanEmbedding.reduce((acc, v) => acc + v * v, 0));
    return meanEmbedding.map(v => v / norm);
  } catch (error) {
    throw new WorkerError('Embedding calculation failed', {
      type: 'embedding_error',
      cause: error
    });
  }
}

// Message handler
onmessage = async (e) => {
  const { command, text } = e.data;
  console.log('Worker received command:', command);

  try {
    if (command === 'init') {
      await init();
    } else if (command === 'infer') {
      if (!isInitialized) {
        throw new Error('Worker not initialized. Call init first.');
      }
      const embedding = await getEmbedding(text);
      postMessage({ type: 'infer-complete', embedding });
    }
  } catch (error) {
    console.error('Worker error:', error);
    postMessage({ type: 'error', error: error.message });
  }
};

// Simple tokenizer definition
function SimpleTokenizer(vocab) {
  this.vocab = vocab;
  this.maxLength = 512;
  this.encode = function(text) {
    const tokens = text.toLowerCase().split(/\s+/);
    const ids = tokens.map(token => this.vocab[token] || this.vocab['[UNK]']);
    const attentionMask = new Array(ids.length).fill(1);
    const tokenTypeIds = new Array(ids.length).fill(0);
    while (ids.length < this.maxLength) {
      ids.push(0);
      attentionMask.push(0);
      tokenTypeIds.push(0);
    }
    if (ids.length > this.maxLength) {
      ids.length = this.maxLength;
      attentionMask.length = this.maxLength;
      tokenTypeIds.length = this.maxLength;
    }
    return { ids, attentionMask, tokenTypeIds };
  };
}
